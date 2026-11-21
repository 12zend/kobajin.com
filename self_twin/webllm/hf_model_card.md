---
language:
  - ja
license: apache-2.0
tags:
  - kobajinai
  - qwen3.5
  - causal-language-model
  - gated-deltanet
  - hybrid-attention
  - lora
  - text-generation
  - conversational
  - japanese
  - webllm
  - mlc-llm
  - webgpu
  - gguf
base_model: Qwen/Qwen3.5-4B
base_model_relation: finetune
pipeline_tag: text-generation
---

# kobajinAI

kobajinAI is a chat model that tries to reply in kobajin's way of speaking. It is a LoRA fine-tune of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (revision [`851bf6e`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)), merged into the base weights.

**Main use: [https://kobajin.com/](https://kobajin.com/).** The site downloads one version's WebLLM bundle from this repository and runs it with WebGPU in the visitor's browser, so replies are generated on the visitor's own device.

kobajinAIは、kobajinの話し方をまねて返事をするチャットモデルです。メインの使い道は [kobajin.com](https://kobajin.com/) です。サイトはこのリポジトリから、選ばれたバージョンのWebLLMバンドルを1つだけダウンロードし、ブラウザのWebGPUで動かします。返事は訪問者の端末の上で生成されます。

## Versions

Every trained version stays published, each in its own folder. The site loads each version from an immutable revision, so its downloads keep working when the repository is reorganized.

| Version | WebLLM bundle | `model_id` | Training data | Character |
|---|---|---|---|---|
| **v2** (site default) | [`v2/` at `1f9287d`](https://huggingface.co/kobajin/kobajinAI/tree/1f9287d837eb311f5d0ab9262240ab2ed362d87c/v2) | `kobajinAI-qwen35-4b-kobajin-clean-v2-1f9287d-q4f16_1-MLC` | Real chats, plus rows that practice using retrieved memories | Casual short replies. Uses memories, but sometimes brings in unrelated ones. |
| v1 | [`v1/` at `8d6dab1`](https://huggingface.co/kobajin/kobajinAI/tree/8d6dab15195505c8c92c2d9ceb99f2551239f2d4/v1) | `kobajinAI-qwen35-4b-kobajin-clean-v1-8d6dab1-q4f16_1-MLC` | Real chats only | Casual short replies. Mostly ignores memories. |
| v3 | [`v3/` at `0bc0bea`](https://huggingface.co/kobajin/kobajinAI/tree/0bc0bea14368ef51162f68ed5c545633ee567f15/v3) | `kobajinAI-qwen35-4b-luna-v3-a2534a2-q4f16_1-MLC` | Mostly synthetic examples written by Luna | Longer, advice-like replies; less like kobajin's real chats. |

The site's default is the version currently judged closest to kobajin (v2 since 2026-11-21). Visitors can pick another version on the site.

## Files

| Path | Format | Used by |
|---|---|---|
| `v1/`, `v2/`, `v3/` | MLC `q4f16_1` WebLLM bundle: `params_shard_*.bin`, `tensor-cache.json`, `mlc-chat-config.json`, tokenizer files | kobajin.com (WebLLM) |
| `gguf/kobajinAI-v1-Q4_K_M.gguf`, `gguf/kobajinAI-v2-Q4_K_M.gguf`, `gguf/kobajinAI-v3-Q4_K_M.gguf` | GGUF `Q4_K_M` | llama.cpp and other GGUF runtimes, for local use. The site does not use them. |

`q4f16_1` stores weights as 4-bit integers with float16 group scales. All three WebLLM bundles reuse WebLLM's prebuilt Qwen3.5-4B library.

v3 was first uploaded to the repository root (revision [`a2534a2`](https://huggingface.co/kobajin/kobajinAI/tree/a2534a20253d54cce925307fbd76086498b09a11)) and moved to `v3/` in [`0bc0bea`](https://huggingface.co/kobajin/kobajinAI/commit/0bc0bea14368ef51162f68ed5c545633ee567f15) without changing any file. kobajin.com still loads v3 from the root of `a2534a2`, so visitors who already downloaded it do not download it again.

The GGUF files were converted from the same merged BF16 checkpoints as the WebLLM bundles (llama.cpp `d81235049`, `convert_hf_to_gguf.py` to BF16, then `llama-quantize Q4_K_M` without an importance matrix). They contain the text model only: Qwen3.5's vision encoder and its multi-token-prediction (MTP) head are not included, so they cannot be used for MTP speculative decoding.

| File | Size | sha256 |
|---|---|---|
| `gguf/kobajinAI-v1-Q4_K_M.gguf` | 2.71 GB | `7baa7ae5f6c7a36147579e0c9ab54e0682641e12b0aad649d523ff81d597341d` |
| `gguf/kobajinAI-v2-Q4_K_M.gguf` | 2.71 GB | `8d9bb50725daa65f44c4cb0a66dd7833693de99187b579a0193a676d839e3a34` |
| `gguf/kobajinAI-v3-Q4_K_M.gguf` | 2.71 GB | `eb893d394dda9ef17f5d984d498597d12b6e12a309f83629456aef0d9a49c430` |

## Use with WebLLM

This is what kobajin.com does (`@mlc-ai/web-llm` 0.2.89). Pin the URL to a revision: WebLLM caches weights by URL.

```js
import { CreateMLCEngine, prebuiltAppConfig } from "@mlc-ai/web-llm";

const base = prebuiltAppConfig.model_list.find((m) => m.model_id === "Qwen3.5-4B-q4f16_1-MLC");
const kobajinAI = {
  ...base,
  model_id: "kobajinAI-qwen35-4b-kobajin-clean-v2-1f9287d-q4f16_1-MLC",
  model: "https://huggingface.co/kobajin/kobajinAI/resolve/1f9287d837eb311f5d0ab9262240ab2ed362d87c/v2/",
  overrides: {
    ...base.overrides,
    // The bundle's template puts "<think>" in the assistant role; with
    // enable_thinking: false the prompt would then differ from training.
    conv_config: {
      ...base.overrides?.conv_config,
      roles: { user: "<|im_start|>user", assistant: "<|im_start|>assistant" },
      stop_token_ids: [248046, 248044],
    },
  },
};

const engine = await CreateMLCEngine(kobajinAI.model_id, {
  appConfig: { ...prebuiltAppConfig, model_list: [kobajinAI] },
});

const reply = await engine.chat.completions.create({
  messages: [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: "11-21 20:15 【参加者A】好きな食べ物は何？" },
  ],
  temperature: 0.72,
  top_p: 1,
  repetition_penalty: 1.12,
  max_tokens: 160,
  extra_body: { enable_thinking: false },
});
```

## Prompt and decoding

The models were trained and evaluated with this system prompt (the site's prompt). The user message is the recent chat log, one line per message, in the form `MM-DD HH:MM 【参加者A】…` (the model's own earlier lines are `【自分】`). The site may add a block of retrieved memory candidates after the system prompt.

```text
次のグループ会話の「自分」として、最後の発言に自然な返事だけをしてください。最新の話題と言語に合わせ、くだけた日本語を基本にします。
軽い相づち、呼びかけ、明確な冗談には短く返してかまいません。質問・相談・理由を求める発言には相づちだけで終わらず、内容に直接答え、理由か具体例を一つ添えて2〜4文で返してください。相手が出した具体的な話題を拾い、関係ない一般論や頼まれていない手順を足さないでください。判断に必要な情報が足りないときは、推測で埋めず一つだけ聞き返してください。
毎回同じ出だしや結論に寄せず、必要なときだけ「僕」を使ってください。「〜すぎて終わった」「たしかに？」「〜でえぐい」のような強めの言い方は、文脈に合うときだけたまに混ぜてください。
あなたはkobajinAIです。自分の名前を聞かれたら「kobajinAI」と答え、別のモデル名を名乗らないでください。返事だけを出し、内部の思考過程や分析は出しません。
今回の質問に関連する本人の記憶がシステム欄にある場合だけ、それを根拠として使ってください。記憶候補の出典IDは回答に表示しません。記憶検索で確認できる記録が見つからない場合、本人の過去の事実や好みを推測で埋めず、そのまま伝えてください。
```

Decoding on the site and in evaluation: temperature `0.72`, top-p `1.0`, no top-k, repetition penalty `1.12` counted over the generated reply only, at most 160 new tokens, thinking disabled. Greedy decoding without a repetition penalty tends to loop.

## Use with llama.cpp

Save the system prompt above as `system_prompt.txt`, then:

```bash
llama-cli --hf-repo kobajin/kobajinAI --hf-file gguf/kobajinAI-v2-Q4_K_M.gguf \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --temp 0.72 --top-p 1.0 --top-k 0 --min-p 0 --repeat-penalty 1.12 \
  -sysf system_prompt.txt
```

llama.cpp's repetition penalty looks at the last 64 tokens, including prompt tokens, so its replies will not match the site exactly.

## Evaluation

Measured on held-out real conversations that were not used in training, with the merged BF16 checkpoints in Transformers, the system prompt above, and the site's decoding. Lower NLL means the model assigns more probability to kobajin's real replies.

| | Base Qwen3.5-4B | v1 | v2 | v3 |
|---|---|---|---|---|
| Token NLL of kobajin's real replies (240 held-out) | 4.826 | **3.617** | 3.639 | 4.332 |
| Relevant memories used (29 cases) | 6 | 0 | **27** | 1 |
| Unrelated memories wrongly used (40 cases) | 3 | **0** | 6 | **0** |
| Fixed memory questions answered with the memory (3) | — | 0 | **2** | 1 (unclear) |

kobajin's real replies have a median of 9.5 characters; v2's generated replies also have a median of 9.5 characters. v3 replies are much longer (median 34 characters) and often contain template phrases from its synthetic data that kobajin does not use.

These numbers come from small test sets and are not a general quality benchmark. They do not show that any version is better than the base model at general tasks.

## Limitations

- **The replies are not kobajin's real words or facts.** All versions sometimes make up plans, situations, and personal details (for example, "今から2時間かけて3D作るぞ！！"). Do not treat anything the model says as a statement by kobajin.
- v2 brings unrelated memories into a reply in about 15% of tested cases and often repeats a memory nearly word for word.
- v1 and v2 often give very short reactions ("ohhh", "www", "えぐい") even when a longer answer would help.
- The 4-bit quantization changes the output slightly from the BF16 checkpoints that were evaluated. On 8 fixed prompts with greedy decoding, v2's WebLLM bundle gave the same reply as Transformers for 4 prompts and v2's GGUF Q4_K_M for 6 (both had the same prompt as Transformers for all 8). v1 and v3 GGUF were only checked to load and reply.
- The models are tuned for casual Japanese group chat. They are not suitable for factual questions, advice that matters, or other languages.

## Training

- Base: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Method: LoRA rank 8, 1 epoch, merged into the base in BF16 with PEFT `merge_and_unload()`
- v1: learning rate 5e-5, 17,830 steps, adapter sha256 `75feda45165454f6cfca0670f0d641f41926da81fb9edc26dfd534cb4d71b4ee`
- v2: learning rate 5e-5, 17,400 steps, adapter sha256 `7a7a7721888c94f50f6f9ca2fba6c65c7f9d286cf0665ea27e5ca53ce91887ba`
- v3: adapter sha256 `0950b43f3f93d7c21938128f017a97a5ff803fd36b8aaf7ed1c6d44b932fe3de`

The training data came from kobajin's own Discord conversations, with other participants anonymized. That data, the memory store, and the evaluation prompts are kept private and are not in this repository. Only the model weights, tokenizer, and configuration are published.

## License

Apache-2.0, the same as the base model.
