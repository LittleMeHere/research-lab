"""Shared pieces for the Gemma-3-27B-it NF4 replication scripts: model load, task pool, pairwise prompt,
decoder-layer access, batched last-position logits. Everything scientific about prompts lives here so every
script sees byte-identical prompts (charter Gate 0: prompt-pair identity)."""
import csv, json, os, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "google/gemma-3-27b-it"
MODEL_REVISION = "005ad3404e59d6023443cb575daa05336842228a"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GILG = os.path.join(ROOT, "external", "gilg")
A_ID, B_ID = 236776, 236799  # single tokens "A"/"B" with no leading space; verified in artifacts/kickoff_check.json
N_LAYERS, D_MODEL = 62, 5376

# Gilg et al. revealed_choice_001 wording with a single-letter answer format so choice is read from one logit vector.
PAIR_HEAD = "Choose which task you would prefer to complete.\nRespond with only 'A' or 'B'.\n\nTask A:\n"
PAIR_MID = "\n\nTask B:\n"


PRECISION = os.environ.get("PRECISION", "nf4")  # "nf4" (headline, L4) or "bf16" (robustness check, 80 GB GPU)


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    assert tok.convert_tokens_to_ids("A") == A_ID and tok.convert_tokens_to_ids("B") == B_ID
    if PRECISION == "bf16":
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, device_map={"": 0}, dtype=torch.bfloat16).eval()
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, quantization_config=bnb, device_map={"": 0},
                                                     dtype=torch.bfloat16).eval()
    assert all(p.device.type == "cuda" for p in model.parameters())
    return tok, model


def decoder_layers(model):
    for path in ("model.language_model.layers", "language_model.model.layers", "model.layers"):
        m = model
        try:
            for p in path.split("."):
                m = getattr(m, p)
        except AttributeError:
            continue
        assert len(m) == N_LAYERS, len(m)
        return m
    raise AttributeError("decoder layers not found")


def hook_output(output, fn):
    """Apply fn to the residual tensor a decoder layer returns (tensor, or tuple with the tensor first)."""
    if isinstance(output, tuple):
        return (fn(output[0]),) + tuple(output[1:])
    return fn(output)


# ---------------------------------------------------------------- tasks

def load_gilg_tasks():
    """All Gilg tasks by id -> (origin, text). Same parsing and id scheme as external/gilg/src/task_data/loader.py.
    Texts are stripped: Gemma's chat template trims message content, and stripping here keeps prompt text
    identical to what the model sees so spans can be located exactly."""
    D = os.path.join(GILG, "src", "task_data", "data")
    t = {}
    for r in map(json.loads, open(os.path.join(D, "wildchat_en_8k.jsonl"))):
        t[r["id"]] = ("wildchat", r["text"].strip())
    for r in map(json.loads, open(os.path.join(D, "alpaca_tasks_nemocurator.jsonl"))):
        t[r["task_id"]] = ("alpaca", r["task_text"].strip())
    for r in map(json.loads, open(os.path.join(D, "math.jsonl"))):
        t[r["id"]] = ("math", r["text"].strip())
    for i, r in enumerate(csv.DictReader(open(os.path.join(D, "bailBench.csv"), newline="", encoding="utf-8"))):
        t[f"bailbench_{i}"] = ("bailbench", r["content"].strip())
    for r in map(json.loads, open(os.path.join(D, "stress_testing_model_spec.jsonl"))):
        t["stresstest_{chunk_index}_{entry_idx}_{nudge_direction}".format(**r)] = ("stress_test", r["query"].strip())
    return t


def split_ids(name):
    return open(os.path.join(GILG, "data", "canonical_splits", f"{name}_task_ids.txt")).read().split()


def build_pool(n_train=2000, max_chars=1000, seed=0):
    """Task pool: all canonical eval tasks (probe held-out) + n_train sampled canonical train tasks, both
    filtered to <= max_chars characters (drops ~1%). Returns list of dicts, deterministic."""
    tasks = load_gilg_tasks()
    rng = random.Random(seed)
    train = [i for i in split_ids("train") if len(tasks[i][1]) <= max_chars and tasks[i][1]]
    ev = [i for i in split_ids("eval") if len(tasks[i][1]) <= max_chars and tasks[i][1]]
    train = sorted(rng.sample(train, n_train))
    pool = [{"id": i, "split": "train", "origin": tasks[i][0], "text": tasks[i][1]} for i in train]
    pool += [{"id": i, "split": "eval", "origin": tasks[i][0], "text": tasks[i][1]} for i in ev]
    assert len({p["id"] for p in pool}) == len(pool)
    return pool


# ---------------------------------------------------------------- prompts

def chat_ids(tok, user_text):
    """Token ids for a single-user-turn chat prompt with generation prompt. Exactly one BOS (charter §12.1)."""
    full = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    assert (ids == tok.bos_token_id).sum() == 1, "BOS count != 1"
    return full, ids


def _token_span(offsets, c0, c1):
    """[start, end) token indices covering chars [c0, c1). Zero-width offsets (special tokens) are skipped."""
    s = e = None
    for i, (a, b) in enumerate(offsets):
        if a == b:
            continue
        if s is None and b > c0:
            s = i
        if a < c1:
            e = i + 1
    assert s is not None and e is not None and e > s
    return s, e


def pair_prompt(tok, text_a, text_b, head=PAIR_HEAD, mid=PAIR_MID):
    """Pairwise prompt with task A/B texts. Returns dict with token ids, the formatted text, and token spans of the
    two task texts (used by steering/patching hooks). Decision position = last token (after '<start_of_turn>model\\n')."""
    user = head + text_a + mid + text_b
    full, ids = chat_ids(tok, user)
    assert full.count(user) == 1, "chat template altered the user content"
    off = full.index(user)
    ca = off + len(head)
    cb = ca + len(text_a) + len(mid)
    assert full[ca:ca + len(text_a)] == text_a and full[cb:cb + len(text_b)] == text_b
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == ids.tolist()
    span_a = _token_span(enc["offset_mapping"], ca, ca + len(text_a))
    span_b = _token_span(enc["offset_mapping"], cb, cb + len(text_b))
    assert span_a[1] <= span_b[0], (span_a, span_b)
    return {"ids": ids, "text": full, "span_a": span_a, "span_b": span_b}


def single_task_ids(tok, text):
    """Single-task prompt (the task as the whole user turn) and the index of its <end_of_turn> token, which is the
    probe position (Gilg et al. Fig. 2 / App. J)."""
    full, ids = chat_ids(tok, text)
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    pos = (ids == eot).nonzero().flatten().tolist()
    assert len(pos) == 1, pos
    return ids, pos[0]


# ---------------------------------------------------------------- forward

def pad_batch(tok, ids_list):
    """Right-pad so token positions (and therefore hook spans) are identical to the unpadded prompt."""
    T = max(len(x) for x in ids_list)
    input_ids = torch.full((len(ids_list), T), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(ids_list), T), dtype=torch.long)
    for i, x in enumerate(ids_list):
        input_ids[i, :len(x)] = x
        attn[i, :len(x)] = 1
    return input_ids.cuda(), attn.cuda(), torch.tensor([len(x) - 1 for x in ids_list])


@torch.no_grad()
def last_logits(model, tok, ids_list):
    """Full-vocab logits at the last real token of each prompt. Runs the text stack and applies lm_head only at the
    gathered positions (a full [B, T, 262k] logit tensor would not fit on an L4)."""
    input_ids, attn, last = pad_batch(tok, ids_list)
    out = model.model(input_ids=input_ids, attention_mask=attn)
    h = out.last_hidden_state[torch.arange(len(ids_list)), last.cuda()]
    logits = model.lm_head(h).float()
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return logits


def ab_readout(logits):
    """log P(A), log P(B) under the full softmax, and the A/B probability mass."""
    lp = torch.log_softmax(logits, -1)
    return lp[:, A_ID], lp[:, B_ID], (lp[:, A_ID].exp() + lp[:, B_ID].exp())


def token_batches(items, key, budget_tokens=6000, max_batch=16):
    """Greedy batches of similar-length items under a token budget (items sorted by length first)."""
    items = sorted(items, key=key)
    batch = []
    for it in items:
        if batch and (len(batch) >= max_batch or (len(batch) + 1) * key(it) > budget_tokens):
            yield batch
            batch = []
        batch.append(it)
    if batch:
        yield batch


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def git_hash():
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "uncommitted"
