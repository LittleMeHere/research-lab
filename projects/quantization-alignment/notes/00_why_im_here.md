# why I'm here  [OWN]

mood: deciding where to focus.

---

option (a) actual ML eng. making AI more capable. physics, music,
  harder reasoning. loved this for months. still love it.

option (b) safety + alignment.

leaning b. why →

- there are ALREADY tons of brilliant engineers on (a). that side will be fine.
- alignment is the bottleneck on whether ANY of (a) ships outside the lab.
  capability without alignment doesn't ship. or it ships and gets pulled.
- was complaining about my own guardrails the other day and the question
  hit: *why aren't YOU doing something about this?*
- so. doing something about it.

position: AI has got to stop being intelligent without judgement.

---

why interp specifically (probably) →

manufacturing background. five-axis CNC, that world.
when something's broken, I open the manual. you can do that with a CNC.
you can't do that with ChatGPT. yet.

mech interp = closest thing to writing the manual.

---

starting with: quantization as an alignment lens.

as small LMs get crushed (int8 → int6 → ternary, maybe binary), what
behaviors degrade first? if refusal-of-harmful-prompts dies before factual
recall, that's a flag worth raising BEFORE someone ships a worse-aligned model.

writeup coming in `quantization_lens.md`.

step one is here - learning the training/quantization pipeline hands-on
with parameter-golf. small model, tight constraints, fast iterations.
step two is running the actual alignment degradation experiments on a
safety-tuned model (gemma, llama-instruct) where refusal behavior actually
exists to measure. need to know how quantization works mechanically
so I can study what it breaks.

---

caveat: not locked in. the field is huge:
  - red teaming
  - evals
  - interp ← leading
  - model organisms
  - governance
  - economics of these systems

scope before commit. ask future-me in a month.
