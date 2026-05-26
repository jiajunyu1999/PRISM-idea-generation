from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset


class JsonlIndexedDataset(Dataset):
    def __init__(self, path: str | Path, limit: int | None = None):
        self.path = Path(path)
        self.offsets: List[int] = []
        with self.path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(pos)
                if limit is not None and len(self.offsets) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_line(self, idx: int) -> Dict[str, Any]:
        with self.path.open("rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode("utf-8")
        return json.loads(line)


class SciJudgePairDataset(JsonlIndexedDataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_length: int = 1024,
        limit: int | None = None,
    ):
        super().__init__(path=path, limit=limit)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._read_line(idx)
        chosen = self.tokenizer(
            row["chosen_text"],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            row["rejected_text"],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen["input_ids"][0],
            "chosen_attention_mask": chosen["attention_mask"][0],
            "rejected_input_ids": rejected["input_ids"][0],
            "rejected_attention_mask": rejected["attention_mask"][0],
            "abs_gap": float(row.get("abs_gap", 0.0)),
            "rel_gap": float(row.get("rel_gap", 0.0)),
            "pair_key": row.get("pair_key", ""),
            "correct_answer": row.get("correct_answer", "A"),
            "meta": row.get("meta", {}),
        }


class SciJudgeEvalPairDataset(JsonlIndexedDataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int = 1024, limit: int | None = None):
        super().__init__(path=path, limit=limit)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._read_line(idx)
        tok_a = self.tokenizer(
            row["text_a"],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        tok_b = self.tokenizer(
            row["text_b"],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "a_input_ids": tok_a["input_ids"][0],
            "a_attention_mask": tok_a["attention_mask"][0],
            "b_input_ids": tok_b["input_ids"][0],
            "b_attention_mask": tok_b["attention_mask"][0],
            "correct_answer": row["correct_answer"],
            "pair_key": row.get("pair_key", ""),
            "meta": row.get("meta", {}),
            "abs_gap": float(row.get("abs_gap", 0.0)),
            "rel_gap": float(row.get("rel_gap", 0.0)),
        }


class TokenizedSciJudgePairDataset(JsonlIndexedDataset):
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._read_line(idx)
        return {
            "chosen_input_ids": torch.tensor(row["chosen_input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(row["chosen_attention_mask"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(row["rejected_input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(row["rejected_attention_mask"], dtype=torch.long),
            "abs_gap": float(row.get("abs_gap", 0.0)),
            "rel_gap": float(row.get("rel_gap", 0.0)),
            "pair_key": row.get("pair_key", ""),
            "correct_answer": row.get("correct_answer", "A"),
            "meta": row.get("meta", {}),
        }


class TokenizedSciJudgeEvalPairDataset(JsonlIndexedDataset):
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._read_line(idx)
        return {
            "a_input_ids": torch.tensor(row["a_input_ids"], dtype=torch.long),
            "a_attention_mask": torch.tensor(row["a_attention_mask"], dtype=torch.long),
            "b_input_ids": torch.tensor(row["b_input_ids"], dtype=torch.long),
            "b_attention_mask": torch.tensor(row["b_attention_mask"], dtype=torch.long),
            "correct_answer": row["correct_answer"],
            "pair_key": row.get("pair_key", ""),
            "meta": row.get("meta", {}),
            "abs_gap": float(row.get("abs_gap", 0.0)),
            "rel_gap": float(row.get("rel_gap", 0.0)),
        }


class SciJudgeSingleDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer, max_length: int = 1024):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tok = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tok["input_ids"][0],
            "attention_mask": tok["attention_mask"][0],
        }


def _pad_stack(seq: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=pad_value)


def pair_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    return {
        "chosen_input_ids": _pad_stack([x["chosen_input_ids"] for x in batch], pad_token_id),
        "chosen_attention_mask": _pad_stack([x["chosen_attention_mask"] for x in batch], 0),
        "rejected_input_ids": _pad_stack([x["rejected_input_ids"] for x in batch], pad_token_id),
        "rejected_attention_mask": _pad_stack([x["rejected_attention_mask"] for x in batch], 0),
        "abs_gap": torch.tensor([x["abs_gap"] for x in batch], dtype=torch.float),
        "rel_gap": torch.tensor([x["rel_gap"] for x in batch], dtype=torch.float),
        "pair_key": [x["pair_key"] for x in batch],
        "correct_answer": [x["correct_answer"] for x in batch],
        "meta": [x["meta"] for x in batch],
    }


def eval_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    return {
        "a_input_ids": _pad_stack([x["a_input_ids"] for x in batch], pad_token_id),
        "a_attention_mask": _pad_stack([x["a_attention_mask"] for x in batch], 0),
        "b_input_ids": _pad_stack([x["b_input_ids"] for x in batch], pad_token_id),
        "b_attention_mask": _pad_stack([x["b_attention_mask"] for x in batch], 0),
        "correct_answer": [x["correct_answer"] for x in batch],
        "pair_key": [x["pair_key"] for x in batch],
        "meta": [x["meta"] for x in batch],
        "abs_gap": [x["abs_gap"] for x in batch],
        "rel_gap": [x["rel_gap"] for x in batch],
    }


def single_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    return {
        "input_ids": _pad_stack([x["input_ids"] for x in batch], pad_token_id),
        "attention_mask": _pad_stack([x["attention_mask"] for x in batch], 0),
    }
