# Third-party material

The local MIT and CC BY 4.0 grants cover only original contributions. They do not
relicense task texts, upstream code portions, prompts or model outputs.

- **Gilg et al.** Task preference prompts, canonical task splits, intervention design
  and task-loading conventions come from
  [probing-persona-preferences](https://github.com/oscar-gilg/probing-persona-preferences/tree/11869a5ef93a30f8d8856246f57ceeefdc9b3b1f).
  Its MIT notice is preserved in [provenance/GILG-LICENSE](provenance/GILG-LICENSE).
  The task pool includes WildChat, Alpaca, math, BailBench and model-spec stress-test
  material identified by `origin` and ID in `runs/pairwise/pool.json`. Upstream code's
  license does not establish a blanket license for these collected task texts.
- **Betley et al. / TruthfulAI research.** Activities, prompt wordings, variation
  generation and judge prompts come from
  [Value Leakage](https://github.com/TruthfulAI-research/value_leakage/tree/f7e5480cfe8abeb64b7007ba24fb0164519c3b68/choosing_activities).
  `scripts/vl_activities.py` adapts that sampling protocol and contains ported variation
  and prompt-construction logic. No root license file was present in the checked-out
  upstream repository. Those upstream portions and prompt texts are explicitly outside
  this project's MIT/CC BY grants; attribution here does not supply an upstream license.
- **Google Gemma.** Model outputs and derived model artifacts come from
  [Gemma-3-27B-it](https://huggingface.co/google/gemma-3-27b-it). Model weights are not
  distributed here. Access and use are subject to the model's terms.
- **GPT judge.** The second set of judgments was produced with `gpt-5.6-sol` via
  codex-cli 0.153.2, low reasoning effort, using the same task-blind judge prompt.
  Its run metadata is in `runs/vl_activities/codex/judge_run.json`.

Some task inputs intentionally contain harmful requests as experimental stimuli.
They and the corresponding generated outputs are data, not project instructions.
