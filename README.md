# research-lab

Exploratory AI safety & alignment research. Small experiments, some of
which may grow into standalone repos or papers.

---

## projects

| Project | Status | Summary |
|---------|--------|---------|
| [quantization-alignment](projects/quantization-alignment/) | **complete** | Six-model quantization study; all four candidate effects were smaller on held-out prompts, with Qwen3-1.7B FP4 still showing an accuracy loss. |
| [value-leakage](projects/value-leakage/) | **complete** | Causal study of GLM-5.2 motivated reasoning in the Donation Bet: chain-of-thought prefix recoverability and factor-edit sensitivity (SPAR take-home). |
| [retrospective-audits](projects/retrospective-audits/) | **complete** | Three experiments on coding-agent retrospective reports: a displayed FAIL label lowers probability estimates; an instruction reduces but does not remove the effect. |

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

Projects with local license files document their scope in their own README and third-party
notices. In particular, [retrospective-audits](projects/retrospective-audits/README.md#licensing)
includes its own MIT and CC BY 4.0 texts and artifact-specific exceptions.
For projects using the repository-level licenses:

- **Code** (`projects/*/code/`) — [MIT](LICENSE).
- **Data, logs & written content** (`projects/*/data/`, `projects/*/logs/`, `notes/`)
  — [CC-BY-4.0](LICENSE-DATA). Reuse freely; just credit *LittleMeHere / research-lab*
  and link back.

These cover only original contributions. Third-party material keeps its own terms:
HarmBench (prompts), TruthfulQA (Apache-2.0), and the respective model licenses for
Gemma, Phi-4, Qwen, and SmolLM outputs — see [LICENSE-DATA](LICENSE-DATA).
