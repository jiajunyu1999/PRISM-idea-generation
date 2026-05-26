# PRISM Idea Generation

Minimal public artifact for PRISM idea generation.

This repository intentionally contains only:

- generated idea outputs,
- GPT-5.5 judge results,
- Galactica SciJudge results,
- simple PRISM generation code.

It excludes paper files, raw experiment folders, local machine paths, and personal filesystem information.

## Files

- `ideas/all_ideas.jsonl`: cleaned generated ideas for all methods/backbones.
- `ideas/all_ideas.csv`: CSV mirror of the generated ideas.
- `ideas/summary_by_backbone_method.csv`: compact idea/token summary.
- `judges/gpt5.5-judge-novelty.jsonl`: GPT-5.5 novelty judge outputs.
- `judges/galactica-judge-impact.jsonl`: Galactica impact judge outputs.
- `judges/galactica/`: Galactica score and pairwise summary tables.
- `code/prism.py`: simple standalone PRISM generation script.

The PRISM method appears as `method=prism` and `method_label=PRISM`.

## Run PRISM

```bash
python code/prism.py \
  --benchmark /path/to/benchmark.jsonl \
  --output outputs/prism_generations.jsonl \
  --backend codex \
  --model gpt-5.4 \
  --limit 3
```

For an OpenAI-compatible API:

```bash
OPENAI_API_KEY=... python code/prism.py \
  --benchmark /path/to/benchmark.jsonl \
  --backend openai \
  --model gpt-5.4
```

Benchmark rows should contain `id`, `task_id`, or `benchmark_id`, plus a problem
field such as `problem_statement` or `problem`.

## Impact Judge
Details see [README.md](./impact_reward/README.md).
