# Agent instructions

Read `PRODUCT.md` before changing this project. The project represents kobajin; do not invent personal memories, preferences, or experiences. Luna-authored synthetic examples may teach response structure and voice cues, but they are not evidence about kobajin's biography or beliefs.

## Model evaluation and Hugging Face release

- Evaluate each candidate against both the currently deployed Hugging Face model and the pinned unadapted base. Keep the prompts, system prompt, and decoding settings the same across candidates.
- Consider a candidate better when held-out conversations show a meaningful improvement in kobajin-like voice or response quality, including relevance, correctness, specificity, naturalness, and useful detail. Validation loss alone is not sufficient. Watch for regressions in short answers, repetition, topic drift, and invented facts.
- When a candidate is demonstrably better than the currently deployed model and has no material regression on the held-out review, merge it into a separate checkpoint, convert it to the WebLLM format configured by the site, and upload the complete converted bundle to `kobajin/kobajinAI` with the authenticated Hugging Face CLI. Keep all shards and tokenizer/config files together.
- The owner may explicitly authorize a clearly labeled experimental release while the base-quality gate remains not passed. Record that authorization and the remaining limitation; do not describe the candidate as better than the base unless evaluation supports that claim.
- Verify the resulting Hugging Face revision and file set after upload. Update evaluation records and release notes with the candidate, base revision, adapter hash, quantization, and comparison results.
- Do not upload Discord exports, private memory files, evaluation prompts containing private conversation text, credentials, or write tokens. Upload only the model bundle and intended public metadata.
- If the candidate does not improve on the deployed model, leave Hugging Face unchanged and record what failed.

## Git milestones

- Work on `main` unless the owner asks for another branch.
- At meaningful, reviewable milestones, commit the focused project changes to `main`. Check the staged diff and exclude model weights, private data, credentials, and generated runtime bundles unless they are explicitly intended source assets.
- Do not push Git commits unless the owner explicitly asks.

## Release workflow

- Preserve the pinned base model and existing releases. Build each candidate in a separate, clearly named directory.
- For WebLLM releases, follow `self_twin/webllm/README.md`; the browser loads MLC-converted weights, not a raw Transformers adapter or merged checkpoint.
- Run the production build and the relevant fixed-prompt comparisons before publishing. Record limitations plainly; a successful upload is not itself evidence of model quality.
