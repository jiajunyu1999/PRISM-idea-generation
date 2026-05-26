from __future__ import annotations

from datetime import datetime
import math
import json
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from train.callbacks import EarlyStopper
from train.losses import build_weight, pairwise_logistic_loss, margin_ranking_loss, score_regularization
from train.metrics import pair_accuracy


class RewardTrainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: Dict,
        output_dir: str | Path,
        tokenizer=None,
        train_sampler=None,
        distributed: bool = False,
        is_main_process: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = -1.0
        self.tokenizer = tokenizer
        self.train_sampler = train_sampler
        self.distributed = distributed
        self.is_main_process = is_main_process
        self.log_path = self.output_dir / "train.log"
        self.metrics_history_path = self.output_dir / "metrics_history.jsonl"
        self.start_epoch = 0
        self.global_step = 0

        grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
        steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
        num_training_steps = cfg["train"]["epochs"] * steps_per_epoch
        scheduler_type = cfg["train"].get("scheduler_type", "linear")
        scheduler_specific_kwargs = {}
        if scheduler_type in {"cosine", "cosine_with_restarts"}:
            scheduler_specific_kwargs["num_cycles"] = float(cfg["train"].get("scheduler_num_cycles", 0.5))
        self.scheduler = get_scheduler(
            name=scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=max(1, int(cfg["train"]["warmup_ratio"] * num_training_steps)),
            num_training_steps=max(1, num_training_steps),
            scheduler_specific_kwargs=scheduler_specific_kwargs or None,
        )
        es_cfg = cfg["train"].get("early_stopping", {})
        self.early_stopper = None
        if es_cfg.get("enabled", False):
            self.early_stopper = EarlyStopper(
                patience=int(es_cfg.get("patience", 3)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
            )

    def _model_for_custom_methods(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _broadcast_should_stop(self, should_stop: bool) -> bool:
        if not self.distributed or not dist.is_initialized():
            return should_stop
        stop_tensor = torch.tensor(int(should_stop), device=self.device)
        dist.broadcast(stop_tensor, src=0)
        return bool(int(stop_tensor.item()))

    def _log(self, message: str) -> None:
        if not self.is_main_process:
            return
        line = f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _append_metrics(self, payload: Dict) -> None:
        if not self.is_main_process:
            return
        with self.metrics_history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load_training_state(self, ckpt_dir: str | Path) -> None:
        state_path = Path(ckpt_dir) / "trainer_state.pt"
        if not state_path.exists():
            self._log(f"resume requested but no trainer_state.pt found under {ckpt_dir}; starting optimizer/scheduler fresh")
            return
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.start_epoch = int(state.get("next_epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.best_val_acc = float(state.get("best_val_acc", self.best_val_acc))
        optimizer_state = state.get("optimizer")
        scheduler_state = state.get("scheduler")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        if self.early_stopper is not None:
            self.early_stopper.load_state_dict(state.get("early_stopper"))
        self._log(
            "loaded trainer state from {} next_epoch={} global_step={} best_val_acc={:.4f}".format(
                ckpt_dir,
                self.start_epoch + 1,
                self.global_step,
                self.best_val_acc,
            )
        )

    def _move(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _is_weighted_epoch(self, epoch: int) -> bool:
        if not self.cfg["loss"]["weighted"]:
            return False
        weighted_start_epoch = self.cfg["train"].get("weighted_start_epoch")
        if weighted_start_epoch is not None:
            return (epoch + 1) >= int(weighted_start_epoch)
        total = self.cfg["train"]["epochs"]
        return epoch >= max(1, total // 3)

    def train(self) -> Dict[str, float]:
        grad_accum = self.cfg["train"]["grad_accum_steps"]
        use_margin = self.cfg["loss"].get("name", "pairwise_logistic") == "margin_ranking"
        should_stop = False
        for epoch in range(self.start_epoch, self.cfg["train"]["epochs"]):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"train epoch {epoch + 1}", disable=not self.is_main_process)
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss_pair_sum = 0.0
            epoch_loss_reg_sum = 0.0
            epoch_step_count = 0

            for step, batch in enumerate(pbar):
                batch = self._move(batch)
                out = self.model(
                    chosen_input_ids=batch["chosen_input_ids"],
                    chosen_attention_mask=batch["chosen_attention_mask"],
                    rejected_input_ids=batch["rejected_input_ids"],
                    rejected_attention_mask=batch["rejected_attention_mask"],
                )
                pos = out["pos_scores"]
                neg = out["neg_scores"]

                weight = None
                if self._is_weighted_epoch(epoch):
                    weight = build_weight(
                        batch["abs_gap"],
                        batch["rel_gap"],
                        w_min=float(self.cfg["loss"].get("weight_w_min", 1.0)),
                        w_max=float(self.cfg["loss"].get("weight_w_max", 5.0)),
                    )

                if use_margin:
                    loss_pair = margin_ranking_loss(
                        pos,
                        neg,
                        margin=float(self.cfg["loss"].get("margin", 1.0)),
                        weight=weight,
                    )
                else:
                    loss_pair = pairwise_logistic_loss(pos, neg, weight=weight)

                loss_reg = score_regularization(pos, neg, coef=float(self.cfg["loss"]["reg_coef"]))
                epoch_loss_pair_sum += float(loss_pair.detach().item())
                epoch_loss_reg_sum += float(loss_reg.detach().item())
                epoch_step_count += 1
                loss = (loss_pair + loss_reg) / grad_accum
                loss.backward()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.global_step % self.cfg["train"]["log_every"] == 0:
                        pbar.set_postfix(
                            {
                                "loss": f"{float(loss_pair.detach().item()):.4f}",
                                "reg": f"{float(loss_reg.detach().item()):.4f}",
                                "weighted": self._is_weighted_epoch(epoch),
                            }
                        )

            metrics = self.evaluate(self.val_loader)
            if self.is_main_process and metrics is not None:
                val_acc = metrics["pair_accuracy"]
                mean_train_loss = epoch_loss_pair_sum / max(epoch_step_count, 1)
                mean_reg_loss = epoch_loss_reg_sum / max(epoch_step_count, 1)
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                weighted_on = bool(self._is_weighted_epoch(epoch))
                self._log(
                    "epoch={} train_loss={:.6f} reg_loss={:.6f} val_acc={:.4f} lr={:.8f} weighted={}".format(
                        epoch + 1,
                        mean_train_loss,
                        mean_reg_loss,
                        val_acc,
                        current_lr,
                        weighted_on,
                    )
                )
                self._append_metrics(
                    {
                        "epoch": epoch + 1,
                        "global_step": self.global_step,
                        "train_loss": mean_train_loss,
                        "reg_loss": mean_reg_loss,
                        "val_pair_accuracy": val_acc,
                        "val_num_examples": metrics["num_examples"],
                        "learning_rate": current_lr,
                        "weighted_loss_on": weighted_on,
                    }
                )
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    if self.cfg["train"].get("save_best_checkpoint", True):
                        self._save_checkpoint("best", next_epoch=epoch + 1)
                    (self.output_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                if self.cfg["train"].get("save_last_checkpoint", True):
                    self._save_checkpoint("last", next_epoch=epoch + 1)
                if self.early_stopper is not None and self.early_stopper.step(val_acc):
                    should_stop = True
                    self._log(f"early stopping triggered at epoch={epoch + 1}")
            should_stop = self._broadcast_should_stop(should_stop)
            if should_stop:
                break

        return {"best_val_acc": self.best_val_acc}

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        if data_loader is None:
            return {"pair_accuracy": 0.0, "num_examples": 0}
        self.model.eval()
        score_a, score_b, labels = [], [], []
        for batch in tqdm(data_loader, desc="eval", leave=False):
            batch = self._move(batch)
            scorer = self._model_for_custom_methods()
            out = scorer.pairwise_compare(
                a_input_ids=batch["a_input_ids"],
                a_attention_mask=batch["a_attention_mask"],
                b_input_ids=batch["b_input_ids"],
                b_attention_mask=batch["b_attention_mask"],
            )
            score_a.extend(out["score_a"].detach().float().cpu().tolist())
            score_b.extend(out["score_b"].detach().float().cpu().tolist())
            labels.extend(batch["correct_answer"])

        if self.distributed and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, {"score_a": score_a, "score_b": score_b, "labels": labels})
            if self.is_main_process:
                merged_score_a, merged_score_b, merged_labels = [], [], []
                for part in gathered:
                    merged_score_a.extend(part["score_a"])
                    merged_score_b.extend(part["score_b"])
                    merged_labels.extend(part["labels"])
                score_a, score_b, labels = merged_score_a, merged_score_b, merged_labels
            else:
                return None

        return {
            "pair_accuracy": pair_accuracy(score_a, score_b, labels),
            "num_examples": len(labels),
        }

    def _save_checkpoint(self, name: str, next_epoch: int) -> None:
        if not self.is_main_process:
            return
        ckpt_dir = self.output_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.backbone.save_pretrained(ckpt_dir / "backbone")
        torch.save(model_to_save.reward_head.state_dict(), ckpt_dir / "reward_head.pt")
        if model_to_save.confidence_head is not None:
            torch.save(model_to_save.confidence_head.state_dict(), ckpt_dir / "confidence_head.pt")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_dir / "tokenizer")
        trainer_state = {
            "next_epoch": next_epoch,
            "global_step": self.global_step,
            "best_val_acc": self.best_val_acc,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "early_stopper": self.early_stopper.state_dict() if self.early_stopper is not None else None,
        }
        torch.save(trainer_state, ckpt_dir / "trainer_state.pt")
