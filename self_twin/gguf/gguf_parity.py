"""Compare a llama-server GGUF model with the Transformers greedy replies of a WebLLM parity run.

usage: python gguf_parity.py <parity-run.json> <compare.json> <merged-checkpoint> <port>
"""
import json, sys, urllib.request
from transformers import AutoTokenizer

run, compare, ckpt, port = sys.argv[1:5]
rows = json.load(open(run))["results"]
ref = {r["prompt"]: r for r in json.load(open(compare))["rows"]}
tok = AutoTokenizer.from_pretrained(ckpt)


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


summary = {"prompts": 0, "prompt_matches_transformers": 0, "greedy_identical": 0, "same_first_60_chars": 0}
out = []
for r in rows:
    msgs = r["messages"]
    hf_prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ll_prompt = post("/apply-template", {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    res = post("/completion", {
        "prompt": hf_prompt, "n_predict": 60, "temperature": 0, "top_k": 1, "repeat_penalty": 1.0,
        "n_probs": 5, "cache_prompt": False, "special": False,
    })
    text = res["content"]
    t_ref = ref[r["prompt"]]["transformers"]
    first = res["completion_probabilities"][0]
    summary["prompts"] += 1
    summary["prompt_matches_transformers"] += ll_prompt == hf_prompt
    n = min(len(text), len(t_ref))
    summary["greedy_identical"] += text[:n] == t_ref[:n]
    out.append({
        "prompt": r["prompt"],
        "llama_cpp_prompt_matches_transformers": ll_prompt == hf_prompt,
        "gguf": text,
        "transformers": t_ref,
        "identical_over_common_length": text[:n] == t_ref[:n],
        "first_token_top5": [(p["token"], round(2.718281828 ** p["logprob"], 4)) for p in first["top_logprobs"]],
    })
print(json.dumps({"summary": summary, "rows": out}, ensure_ascii=False, indent=1))
