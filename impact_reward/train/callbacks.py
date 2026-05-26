from __future__ import annotations


class EarlyStopper:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_steps = 0

    def step(self, value: float) -> bool:
        if self.best is None or value > (self.best + self.min_delta):
            self.best = value
            self.bad_steps = 0
            return False
        self.bad_steps += 1
        return self.bad_steps >= self.patience

    def state_dict(self) -> dict:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "bad_steps": self.bad_steps,
        }

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.best = state.get("best", self.best)
        self.bad_steps = state.get("bad_steps", self.bad_steps)
