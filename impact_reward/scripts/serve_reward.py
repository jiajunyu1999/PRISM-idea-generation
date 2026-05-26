from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import uvicorn

from data.formatter import format_idea_input
from models.io import load_reward_model


class ScoreRequest(BaseModel):
    field: str
    subfield: str | None = ""
    title: str
    description: str


class CompareRequest(BaseModel):
    idea_a: ScoreRequest
    idea_b: ScoreRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve reward model API")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_reward_model(args.ckpt, device=device)

    app = FastAPI(title="SciJudge Reward Service")

    def _score_text(text: str) -> float:
        tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.no_grad():
            score = model.score(tok["input_ids"], tok["attention_mask"])
            if isinstance(score, dict):
                score = score["score"]
        return float(score.detach().cpu().item())

    @app.post("/score")
    def score(req: ScoreRequest):
        text = format_idea_input(req.model_dump())
        reward = _score_text(text)
        normalized = float(torch.sigmoid(torch.tensor(reward)).item())
        return {
            "reward": reward,
            "normalized_reward": normalized,
            "confidence": 0.5,
        }

    @app.post("/compare")
    def compare(req: CompareRequest):
        text_a = format_idea_input(req.idea_a.model_dump())
        text_b = format_idea_input(req.idea_b.model_dump())
        score_a = _score_text(text_a)
        score_b = _score_text(text_b)
        p = float(torch.sigmoid(torch.tensor(score_a - score_b)).item())
        return {
            "score_a": score_a,
            "score_b": score_b,
            "p_a_beats_b": p,
            "winner": "A" if score_a > score_b else "B",
        }

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
