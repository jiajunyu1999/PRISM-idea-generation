#!/usr/bin/env python3
"""Minimal PRISM idea-generation code.

PRISM The method generates one grounded research idea from
four fixed risk-validated views; optionally, a caller can preselect a subset of
view IDs and pass them through --selected-views.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


METHOD = "prism"

PRISM_VIEWS = [
    {
        "view_id": "V1",
        "name": "Novelty Gap with False-Positive Rejection",
        "focus": (
            "Frame the idea around an unmet gap and explicitly reject "
            "false-positive, trivial, or already-solved explanations."
        ),
    },
    {
        "view_id": "V2",
        "name": "Mechanism or Intervention",
        "focus": "Specify the concrete mechanism, algorithm, intervention, or scientific move that solves the gap.",
    },
    {
        "view_id": "V3",
        "name": "Risk and Uncertainty Calibration",
        "focus": "Identify uncertainty, confounds, failure modes, and how the idea calibrates or controls them.",
    },
    {
        "view_id": "V4",
        "name": "Minimal Validation Benchmark",
        "focus": "Design the smallest convincing experiment, benchmark, or prospective test with measurable success criteria.",
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: expected JSON object rows")
                rows.append(value)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def task_id(row: Mapping[str, Any]) -> str:
    value = row.get("benchmark_id") or row.get("task_id") or row.get("id")
    if not value:
        raise KeyError("task row must contain benchmark_id, task_id, or id")
    return str(value)


def task_context(row: Mapping[str, Any]) -> str:
    rubric = row.get("evaluation_rubric", row.get("constraints", ""))
    if isinstance(rubric, list):
        rubric = "; ".join(str(item) for item in rubric)
    lines = [
        f"Task ID: {task_id(row)}",
        f"Domain: {row.get('domain', '')}",
        f"Problem: {row.get('problem_statement', row.get('problem', ''))}",
    ]
    if row.get("background_context"):
        lines.append(f"Background: {row['background_context']}")
    if row.get("expected_output_type"):
        lines.append(f"Expected output: {row['expected_output_type']}")
    if rubric:
        lines.append(f"Evaluation rubric: {rubric}")
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def selected_views(view_ids: list[str]) -> list[dict[str, str]]:
    by_id = {view["view_id"]: view for view in PRISM_VIEWS}
    if not view_ids:
        return PRISM_VIEWS
    missing = [view_id for view_id in view_ids if view_id not in by_id]
    if missing:
        raise ValueError(f"unknown PRISM view IDs: {missing}")
    return [by_id[view_id] for view_id in view_ids]


def build_prompt(task: Mapping[str, Any], views: list[dict[str, str]], min_words: int, max_words: int) -> str:
    return f"""
Generate one strong PRISM research idea for this task. Return only valid JSON.

Task:
{task_context(task)}

PRISM views:
{json.dumps(views, ensure_ascii=False, indent=2)}

Instructions:
- Use the PRISM views as constraints, but do not list them as sections.
- Keep the proposal grounded in the exact task.
- Preserve novelty gap, concrete mechanism, risk handling, and minimal validation.
- Reject likely false-positive, trivial, or already-solved explanations where relevant.
- Do not mention prompts, judges, scores, or internal benchmarking.

Output requirements:
- One English abstract-style paragraph.
- Target length: {min_words}-{max_words} words.
- First sentence starts with "We propose" and names a concrete method.
- No headings, bullet points, citations, or labels.

JSON shape:
{{
  "task_id": "{task_id(task)}",
  "method": "{METHOD}",
  "selected_views": ["V1", "V2", "V4"],
  "idea": "{min_words}-{max_words} word paragraph..."
}}
""".strip()


def run_codex(prompt: str, model: str, timeout: int, reasoning_effort: str) -> str:
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as handle:
        output_path = Path(handle.name)
    cmd = [
        "codex",
        "exec",
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-m",
        model,
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(output_path),
        prompt,
    ]
    try:
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])
        return output_path.read_text(encoding="utf-8").strip()
    finally:
        output_path.unlink(missing_ok=True)


def run_openai_compatible(prompt: str, model: str, timeout: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL"))
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You generate rigorous research ideas and return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/prism_generations.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", choices=["codex", "openai"], default="codex")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--min-words", type=int, default=285)
    parser.add_argument("--max-words", type=int, default=315)
    parser.add_argument("--selected-views", default="", help="Comma-separated PRISM view IDs. Empty uses all four views.")
    args = parser.parse_args()

    view_ids = [item.strip().upper() for item in args.selected_views.split(",") if item.strip()]
    views = selected_views(view_ids)
    tasks = read_jsonl(args.benchmark)
    if args.limit:
        tasks = tasks[: args.limit]

    for task in tasks:
        prompt = build_prompt(task, views, args.min_words, args.max_words)
        if args.backend == "codex":
            raw = run_codex(prompt, args.model, args.timeout, args.reasoning_effort)
        else:
            raw = run_openai_compatible(prompt, args.model, args.timeout)
        parsed = parse_json_object(raw)
        idea = str(parsed.get("idea", "")).strip()
        if not idea:
            raise ValueError(f"{task_id(task)}: empty idea")
        append_jsonl(
            args.output,
            {
                "task_id": task_id(task),
                "method": METHOD,
                "method_label": "PRISM",
                "selected_views": parsed.get("selected_views") or [view["view_id"] for view in views],
                "idea": idea,
                "word_count": word_count(idea),
                "model": args.model,
            },
        )
        print(f"[done] {task_id(task)} words={word_count(idea)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
