# research-lab

Exploratory AI safety & alignment research. Small experiments, some of
which may grow into standalone repos or papers.

---

## projects

| Project | Status | Summary |
|---------|--------|---------|
| [quantization-alignment](projects/quantization-alignment/) | **data complete** | Does quantization degrade safety faster than capability? 6 models × 4 quant levels, ~6K inferences. |

---

## structure

```
projects/
  <project-name>/
    README.md           — what it is, what it found
    requirements.txt    — dependencies
    code/               — experiment scripts
    data/               — raw results
    notes/              — writeups, analysis
    logs/               — execution logs
```

Each project is self-contained. If something graduates to a paper or
standalone tool, it gets extracted into its own repo.

## tools

Built with a mix of: Claude Code, Gemini CLI, Jules, Codex, Vertex AI.
Stress-testing them against my ideas and each other.

## license

[MIT](LICENSE) — code is free. If you use the data or findings,
a citation is appreciated.
