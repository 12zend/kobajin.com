# Product: kobajin.com

## What it is

`kobajin.com` is a conversational self-twin project that aims to reproduce kobajin's way of speaking and responding. It uses a locally run WebLLM model in the visitor's browser, with LoRA training and carefully bounded memory retrieval to make the replies more like kobajin.

## Product goal

Make replies feel like kobajin while remaining useful, coherent, context-aware, and accurate. Preserve the casual, fragmented rhythm of ordinary chat, but provide substantial answers when someone asks for an explanation, opinion, plan, or help.

**文字の質はあればあるだけ良い。** Improve the quality of the text wherever possible: answer the actual point, use relevant context, explain the reason, and add useful specifics. Do not treat shortness as a goal for its own sake or pad a reply with empty words. A brief reaction is right when the conversation only calls for one.

## Product experience

- The site is a static web application. The Qwen3.5-based WebLLM model runs through WebGPU in the visitor's browser.
- The public Hugging Face repository `kobajin/kobajinAI` stores the MLC-converted model bundle the browser downloads.
- Replies should use casual Japanese when the conversation does, preserve natural reactions and message breaks, and avoid repeating the same opening or advice.
- Stronger phrases such as 「〜すぎて終わった」「たしかに？」「〜でえぐい」 can appear occasionally when they fit the moment.
- Verified personal memories may support factual answers. If a memory is missing or unclear, the twin must say so instead of making up an autobiographical detail.

## Data and model principles

- Keep private conversation exports and personal-memory stores local. Never upload them as model artifacts.
- Synthetic training examples can teach helpful response patterns and evidence-backed style cues; they must not be presented as kobajin's real memories, opinions, or preferences.
- Judge candidates on held-out conversation quality and kobajin-like voice, not training loss alone. Check relevance, correctness, useful detail, naturalness, repetition, and topic drift.
- Follow `AGENTS.md` for model promotion, Hugging Face uploads, and Git milestones.
