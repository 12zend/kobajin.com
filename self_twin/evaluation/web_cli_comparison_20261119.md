# Web / CLI output comparison — 2026-11-19

## What was compared

The CLI references are the stored Luna v3 synthetic held-out generations in `generations-r8-luna-synthetic-heldout-v3-system-v2-sampled.jsonl`. They use the shared system prompt, the pinned BF16 base plus v3 adapter, `temperature=0.72`, `top_p=1.0`, `repetition_penalty=1.12`, and `max_new_tokens=160`.

The Web outputs below came from three fresh chats in the local WebLLM app before the cache fix. The app used the same source system prompt and sampling values, but it used the mutable `resolve/main/` URL, q4f16_1 weights, and `max_tokens=384`. The browser may have reused model shards cached under that mutable URL, so these examples are observational rather than a controlled comparison of the same revision and random seed.

## Examples

| Synthetic prompt | CLI v3 reference | Web before fix |
| --- | --- | --- |
| 頼まれごと断ったら気まずくなった | その場だけならまだしも、「後でどう言うべきかなんて考えずに先を見ないまま話して終わりそう」って心配になる。全部一回やるのは難しいから、「後でいいよと伝えられるか試す」まで小さくしよ。 | 断り方だけ変えても良し悪しが別れる。「その頼まれごとの理由が」は、相手の期待やあなたの負担を両方考える必要がありそうだから、今回は省略して「自分ならこうするよ！くらいでいいかも！？」 |
| コンソールに警告が大量に出てる | 全部読まずに「失敗した操作とエラーメッセージ」を先に比べよ。僕ならまずそれだけ見たいな。 | わっ！複数件で書き換えるだけかな。 |
| ゲームの難易度が急に上がる | あーそいうことあるよね、最初のほうだけ見ると難易度調整されてる気がして。あと序盤の失敗と後の失敗が同じ原因じゃないから、両方見てたい。 | あーね。難易度どうすん？ |

In these samples, the CLI responses more often add a concrete angle while the Web replies tend to stop at a short reaction or a generic suggestion. The CLI references also contain awkward generated text, so they are not proof that the CLI fully captures kobajin's voice.

## Changes made

- The Web app now pins the model URL to HF revision `a2534a20253d54cce925307fbd76086498b09a11` and uses a release-specific WebLLM `model_id`. WebLLM caches weights by model URL; a mutable `resolve/main/` URL can therefore keep serving stale shards after an upload.
- Web generation now uses a 160-token output limit to match the CLI evaluation cap. The shared system prompt, temperature, top-p, and repetition penalty remain aligned.
- A post-fix generation could not be completed during this comparison: the pinned browser download reached 263 MB (12%) of the approximately 2.2 GiB bundle after 152 seconds. The production build succeeds; output quality with the pinned cache key still needs a completed browser run.
