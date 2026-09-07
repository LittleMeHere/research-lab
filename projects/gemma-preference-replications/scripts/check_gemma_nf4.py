"""Kickoff smoke test for Gemma-3-27B-it in NF4.

Checks the Gemma-specific invariants from charter §12.1 before any experiment code exists:
  1. exactly one BOS after apply_chat_template (Gemma's template inserts <bos>; tokenizing again must not add a second)
  2. label token ids for A/B, with and without a leading space (the model answers after "<start_of_turn>model\n",
     so the no-space variant is the one that matters at the decision position)
  3. one greedy chat completion so the load path is visibly sane
  4. peak VRAM after load and after generation
Writes artifacts/kickoff_check.json (raw) and prints the same.
"""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "google/gemma-3-27b-it"
out = {"model": MODEL, "precision": "nf4-bnb", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

tok = AutoTokenizer.from_pretrained(MODEL)
bos = tok.bos_token_id
out["bos_token_id"] = bos

messages = [{"role": "user", "content": "Which is larger, 7 or 12? Answer with just the number."}]
prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out["chat_template_text"] = prompt_text

# Two tokenization paths; both must yield exactly one BOS.
ids_a = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
if not torch.is_tensor(ids_a):  # newer transformers return a BatchEncoding
    ids_a = ids_a["input_ids"]
ids_b = tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
ids_c = tok(prompt_text, return_tensors="pt")["input_ids"]  # default add_special_tokens=True: the double-BOS trap
out["n_bos"] = {
    "apply_chat_template(tokenize=True)": int((ids_a == bos).sum()),
    "tok(text, add_special_tokens=False)": int((ids_b == bos).sum()),
    "tok(text) default": int((ids_c == bos).sum()),
}
out["first_tokens"] = tok.convert_ids_to_tokens(ids_a[0, :6].tolist())
assert out["n_bos"]["apply_chat_template(tokenize=True)"] == 1, out["n_bos"]
assert out["n_bos"]["tok(text, add_special_tokens=False)"] == 1, out["n_bos"]
assert ids_a.tolist() == ids_b.tolist(), "chat-template tokenization paths disagree"

# Label token ids. encode() without specials; a label must be a single token to be read off one logit vector.
labels = {}
for s in ["A", "B", " A", " B", "(A", "(B", "**A", "**B"]:
    e = tok.encode(s, add_special_tokens=False)
    labels[s] = {"ids": e, "tokens": tok.convert_ids_to_tokens(e)}
out["label_tokens"] = labels
for s in ["A", "B", " A", " B"]:
    assert len(labels[s]["ids"]) == 1, (s, labels[s])

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16)
model.eval()
out["load_seconds"] = round(time.time() - t0, 1)
out["n_layers"] = model.config.get_text_config().num_hidden_layers
out["d_model"] = model.config.get_text_config().hidden_size
out["vram_after_load_GB"] = {"allocated": round(torch.cuda.memory_allocated() / 2**30, 2),
                             "peak": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
# device_map={"":0} must put every parameter on the GPU; an offloaded module would silently run on CPU.
assert all(p.device.type == "cuda" for p in model.parameters()), "some parameters not on cuda"

with torch.no_grad():
    inp = ids_a.to(0)
    logits = model(inp).logits[0, -1].float()
    top = torch.topk(logits, 5)
    out["next_token_top5"] = [(tok.convert_ids_to_tokens(int(i)), round(float(v), 2)) for v, i in zip(top.values, top.indices)]
    pA, pB = (logits.softmax(-1)[labels[s]["ids"][0]].item() for s in ("A", "B"))
    out["p_A_p_B_at_decision_pos"] = [round(pA, 5), round(pB, 5)]
    gen = model.generate(inp, max_new_tokens=20, do_sample=False)
out["completion"] = tok.decode(gen[0, inp.shape[1]:], skip_special_tokens=False)
out["vram_after_generate_GB"] = {"allocated": round(torch.cuda.memory_allocated() / 2**30, 2),
                                 "peak": round(torch.cuda.max_memory_allocated() / 2**30, 2)}

os.makedirs("artifacts", exist_ok=True)
json.dump(out, open("artifacts/kickoff_check.json", "w"), indent=2)
print(json.dumps(out, indent=2))
