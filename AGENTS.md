# Agent instructions

Read `PRODUCT.md` before changing this project. The project represents kobajin; do not invent personal memories, preferences, or experiences. Luna-authored synthetic examples may teach response structure and voice cues, but they are not evidence about kobajin's biography or beliefs.

## Model evaluation and Hugging Face release

- Every trained version is published. The owner decided on 2026-11-21 that each finished model (v1, v2, v3, and later v4 and onward) is uploaded to `kobajin/kobajinAI` and can be chosen on the site. Evaluation no longer decides whether a version is uploaded. It decides what the release notes say about it and which version is the site's default.
- Evaluate each new version before publishing it. Compare it with the site's current default model and with the pinned unadapted base, using the same prompts, system prompt, and decoding settings. Validation loss alone is not sufficient. Look at held-out conversations for kobajin-like voice, relevance, correctness, specificity, naturalness, and useful detail. Watch for regressions in short answers, repetition, topic drift, and invented facts.
- Upload each version as its own complete WebLLM bundle in its own folder of `kobajin/kobajinAI` (for example `v2/`), with the authenticated Hugging Face CLI. Keep all shards and tokenizer/config files together.
- Add each published version to the site's model list in `src/models.js`, pinned to its immutable revision and folder, with a versioned `model_id` and a plain description. The site's default is the version named in `src/closest_model.json`, the one currently judged closest to kobajin. Change that file only when evaluation shows another version is closer to kobajin than the current one with no material regression, or when the owner asks. Record the date, who decided, and the evidence in it.
- Record each version's evaluation plainly, including its known limitations. Do not describe a version as better than the base or another version unless evaluation supports that claim.
- Verify the resulting Hugging Face revision and file set after upload. Update evaluation records and release notes with the version, base revision, adapter hash, quantization, Hugging Face revision, and comparison results.
- Do not upload Discord exports, private memory files, evaluation prompts containing private conversation text, credentials, or write tokens. Upload only the model bundle and intended public metadata.

## Git milestones

- Work on `main` unless the owner asks for another branch.
- Every agent commits to `main`, including parallel or background sessions. Do not leave finished work on a separate branch such as a `claude/*` worktree branch. If a session has to work in its own worktree, bring its commits onto `main` when the work is done, by fast-forward or cherry-pick, and then remove the temporary branch and worktree.
- At meaningful, reviewable milestones, commit the focused project changes to `main`. Check the staged diff and exclude model weights, private data, credentials, and generated runtime bundles unless they are explicitly intended source assets.
- Do not push Git commits unless the owner explicitly asks.

## Release workflow

- Preserve the pinned base model and existing releases. Build each candidate in a separate, clearly named directory.
- For WebLLM releases, follow `self_twin/webllm/README.md`; the browser loads MLC-converted weights, not a raw Transformers adapter or merged checkpoint.
- Pin every WebLLM model URL to an immutable Hugging Face revision and give each release a versioned `model_id`. If `VITE_MODEL_URL` is set for local bundle testing, verify that it serves the intended bundle. WebLLM caches by model URL, so replacing files behind `resolve/main/` can leave old weights in the browser.
- The site opens on a start screen, not on the timeline, and shows nothing generated until a kobajinAI version is running. There is no mock fallback for visitors, so a screenshot of the site never shows canned text (owner decision, 2026-11-21). The start screen says that the device needs several GB of free storage, warns that 8 GB machines may be slow or stop, and offers a download button for the default version (the closest one, `src/closest_model.json`).
- The site downloads and caches a model only for the version a visitor picks, and loads only that one version into memory: when the visitor presses the button, or, on later visits, when the page opens if that visitor already ran it and it is still saved in the browser (the owner asked for this on 2026-11-21). A first-time visitor downloads nothing until they press the button. Stopping, an engine error, or another tab taking the model returns to the start screen. Never load several versions at once.
- Run the production build and the relevant fixed-prompt comparisons before publishing. Record limitations plainly; a successful upload is not itself evidence of model quality.
