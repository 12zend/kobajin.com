# 学習レポート

## 状態

2026-11-19: Ownerの要望に基づき、従来の合成データを置き換えた。実ログで確認できたScratch、ソフトウェア、3D、Web、映像、LLM、共同制作の話題を使い、個人の過去・好み・信条を含まない汎用シナリオを200件作成。各シナリオを診断、説明、最初の手順、懸念への応答、確認方法、代案、平易な説明、切り分け、追加質問、短い要約の10 intentへ展開した。train 1,800 / validation 200、シナリオ単位で分離。2,000件すべて入力と回答がユニーク。返答長は中央値87文字（短い要約intentは62文字）で、各intentの長さもmanifestへ記録。Discord本文は転載していない。`training/build_synthetic_dataset.py` と `training/synthetic_case_bank.jsonl` から再生成可能。これは合成データの改稿記録であり、学習・モデル評価はまだ実施していない。

2026-11-19: Ownerの要望に基づき、Lunaが合成例1,120件を新規作成した。257シナリオから診断・懸念への返答・具体策・成功基準の4種類を単独入力として生成し、説明・短文例92件を加えた。train 1,008 / validation 112で、各splitはシナリオ単位で分離。入力と返答は全件ユニーク。最初の抜き取り確認で一部の返事が懸念や話題に合わなかったため、回答の役割を分け、懸念への返事を見直した。合成データに過去のassistant発言を連ねず、本人の事実や価値観も作っていない。依頼にあった言い方は14件（1.25%）だけに入れた（「すぎて終わった」4件、「たしかに？」5件、「えぐい」5件）。口調はプロフィールで確認できるくだけた日本語、短い投稿分割、断定を和らげる語を参考にした。`build_synthetic_dataset.py` で再生成できる。

先に起動した92例版pilotはownerの追加指示を受けて43/1,000 stepで停止し、checkpointは作成されていない。この不完全なrunは採用しない。新しいrank 8 pilotは1,000例・1 epoch・LR `1e-5`、合成例35%、通常例488件、低比重reaction162件で完了。train lossは3.718、validation lossは3.547。adapterは`adapter/experiment-r8-luna-synthetic-1000-v2/adapter/`に別保存した。WebLLM側はtemperature 0をやめ、依頼内容によって短文と具体的な説明を使い分けるsystem instructionへ変更済み。

2026-11-19のrank 8 v2評価: 実会話35件をWebLLM相当のsampling（temperature 0.72、top_p 1.0、repetition penalty 1.12、seed 42）でbase/LoRA比較した。会話返信31件の補助char ROUGE-L-F1はbase `0.0941`、LoRA `0.0676`。LoRAは出力中央値10文字、5文字以下14/35、100文字超3/35、連続反復span検出0件。既知記憶はLoRA+RAGで3/3の出典を取得し、未知記憶への意図しない検索は0件。合成held-out validationから20件（4応答意図×5件）を別評価したところ、char ROUGE-L-F1はbase `0.1520`、LoRA `0.2734`、応答中央値49文字、5文字以下0件、連続反復span検出0件。ただし、具体策や話題を取り違えた例が残る。合成応答形式への追従は上がったが、実会話への一般化は不十分と判断し、品質ゲートを通過扱いにしない。出力は`evaluation/generations-r8-luna-synthetic-1000-v2-sampled.jsonl`、集計は`evaluation/benchmark-r8-luna-synthetic-1000-v2-sampled.json`、盲検シートは`evaluation/subjective_review-r8-luna-synthetic-1000-v2-sampled.md`に保存した。merge・公開更新は行っていない。

2026-11-19のrank 8 v3: 合成例比率を70%に上げ、1,000例（合成700・低比重reaction150・通常例150）、1 epoch、LR `1e-5`で学習。train loss `2.9548`、validation loss `3.2378`。MPSの検証時に既定eval batch sizeが大きく遅くなることを確認したため、`per_device_eval_batch_size=1`と`prediction_loss_only=True`を追加し、最終runでは312 validation rowsを約139秒で評価した。adapterは`adapter/experiment-r8-luna-synthetic-1000-v3/adapter/`へ保存。

評価のsystem promptをWebLLM本番と共有する`inference/system_prompt.txt`へ統一し、同じ35件・20件・sampling条件（temperature `0.72`、top_p `1.0`、repetition penalty `1.12`、seed 42 / 73）でbase、現行HF候補、v2、v3を比較した。実会話35件の会話返信char ROUGE-L-F1はbase `0.1213`、現行HF rank8/5,000候補 `0.0492`、v2 `0.0850`、v3 `0.0999`。v3の返答中央値は現行公開候補の`5`文字から`34`文字へ増え、5文字以下は`18/35`から`3/35`。合成held-out20件では現行HF `0.0746`、v2 `0.2003`、v3 `0.2436`。v3は現行公開候補より改善したためownerが実験版公開を明示し、別checkpointへmerge後q4f16_1へ変換してHugging Faceへアップロードした（commit [`a2534a2`](https://huggingface.co/kobajin/kobajinAI/commit/a2534a20253d54cce925307fbd76086498b09a11)）。baseにはまだ届かず、品質ゲートは未通過のまま。出力は`evaluation/generations-r8-luna-synthetic-1000-v3-system-v2-sampled.jsonl`、`evaluation/generations-r8-balanced-5000-v3prompt-sampled.jsonl`および対応する合成held-out files。集計は対応する`benchmark-*.json`、盲検シートは`subjective_review-r8-luna-synthetic-1000-v3-system-v2-sampled.md`と`subjective_review-synthetic-heldout-v3-system-v2-sampled.md`。

rank 8 / 5,000例の主観レビューを受領した。35件の全体平均は2.66/5、会話31件は2.45/5で、17件にループ、9件に短さの指摘があった。既知記憶3件は4.67/5。調査で従来のベンチマークと推論コードがTokenizerのEOS IDを生成停止条件に渡していないことが判明した。Qwen3.5 configにもEOS IDがないため、旧生成は128-token上限まで続き、会話role漏れと反復を含んでいた。共通生成ヘルパーで`<|im_end|>`停止と次ターン境界ガードを実装し、同じ35件を4候補すべて再評価した。role漏れは0件になったが、rank 8 / 5,000例は5ケースに反復、24/35件が5文字以下。ほかのrankも反復が11〜14件あり、短さと反復のバランスを解決できていない。TRLのtraining templateはassistant終端tokenをloss maskに含むことを確認したため、EOS停止だけを理由に再学習しない。旧レビュー点数は旧出力に対する暫定値で、修正後の簡易シートを別途作成した。品質ゲートは未通過、マージしない。

初回rank 8 pilotは旧データ版の680/2,000 stepsで停止し、話題転換59か所を分割したデータでrank 8 / rank 16のpilot（各1,000例×2 epoch）を実行した。生成時に長い低多様性反復がrank 8で15/35件、rank 16で18/35件起き、未知記憶の誤答もあったため不合格だった。調査で、このrunは既定のthinking prefixで学習しながら推論では`enable_thinking=false`を使用していた不一致を発見したため、これらのadapterを選定対象から外す。修正版は学習・推論ともに`enable_thinking=false`とした。

修正版rank 8・250例×1 epoch・LR 1e-5のpilotは完了した。mask preflightは通過、validation lossは4.194。LoRA出力中央値は3文字で、held-out本人らしさの改善は確認できなかった。整合済み設定の通常サンプリング1,000例×1 epoch・LR 1e-5でもvalidation loss 4.017、出力中央値2文字、35件中28件が5文字以下、100文字超は1件のみとなり不合格。低比重reactionを25%にしたrank 8 / 1,000例pilotはvalidation loss 3.990、出力中央値10文字、35件中14件が5文字以下、100文字超9件中8件が低多様性反復だった。対応するrank 16 / 1,000例pilotも完了し、train loss 4.012、validation loss 3.992。両rankはseed、1,000例（通常750 / 低比重250）、LR `1e-5`、1 epoch、thinking無効、template hash、評価35件を揃えた。全34参照回答（会話31 + 記憶3）のchar ROUGE-L-F1はrank 8 `0.0449`、rank 16 `0.0447`、LoRA+RAGは`0.0792` / `0.0797`、base `0.0912`。会話31件のみではLoRA rank 8 `0.0304`、rank 16 `0.0310`、base `0.0901`。rank 16に増やしてもvalidation・held-out style similarityの改善はなかった。

rank 8 / 5,000例pilotはtrain loss 3.784、validation loss 3.791で完了した。全34参照回答（会話31 + 記憶3）のchar ROUGE-L-F1はbase `0.0912`、LoRA `0.0636`、LoRA+RAG `0.0728`。会話31件だけではbase `0.0901`、LoRA / LoRA+RAG `0.0628`。LoRA応答中央値は18文字（5文字以下14/35、100文字超14/35、長い低多様性応答2/35）。低比重reactionを10%にしたrank 8 / 1,000例runはtrain loss 4.114、validation loss 3.998。応答中央値44文字、5文字以下7/35、長い低多様性応答11/35。会話31件のchar ROUGE-L-F1はbase `0.0901`、LoRA `0.0498`、LoRA+RAG `0.0498`。反応比率10%で短文偏重は弱まった一方、反復指標が悪化し、会話スタイルはbaseを下回った。

評価レポートでは総合char ROUGE-L-F1が記憶事実3件を含むため、会話31件のみの値を分けて記録する。全34参照回答の数値を文体指標と誤解しないよう、benchmark JSONにもscope別の指標と参照件数を追加した。rank 8 / 5,000例、rank 16 / 1,000例、rank 8 / 低比重10%の盲検レビューシートを生成済み。自動評価は主観評価の代用ではなく、full runとmergeは品質ゲート通過まで行わない。

## 学習前に判明した条件

- 学習データはtrain 22,818サンプル、validation 2,791、test 2,842。本人の3 IDを統合し、本人の連続投稿は23,861ペア確認済み。
- 実行環境はApple M6 Mac mini、統合メモリ32GB。project venvにはPyTorch 2.15.0 / Transformers 5.23.0 / PEFT 0.21.3 / TRL 1.18.1を導入した。CUDAはなく、GPUアクセスを許可したプロセスからMPSとBF16行列演算が使えることを確認した。
- この実験では、Unsloth経路を使わず、Transformers / PEFT / TRLのMPS経路を実際にロード・学習できることを確かめて使っている。公式post-trained `Qwen/Qwen3.5-4B` のSafetensorsをローカル取得し、revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` に固定した。BF16、無量子化、Safetensors形式を確認済み。
- データはこの公式revisionのTokenizerとChat Templateで再計数した。学習用会話全体でtrain 6,996,616 token / validation 858,124 / test 876,157。trainの系列長中央値303、95%点375、最大10,681 token。
- UnslothのQwen3.5ガイドはTransformers v5を要求し、4B BF16 LoRAのVRAM目安を約10 GBとしている。また、4bit QLoRAは量子化差のため推奨していない。本実験環境はApple SiliconのMPS統合メモリなので、UnslothのVRAM目安を直接適用せず、実測したTransformers MPS経路を使う。
- 現行ローカルChat Templateにgeneration maskマーカーがないが、TRLのQwen3.5 training templateを適用した。実collatorのlabelsがassistant generation maskと一致し、prompt全tokenが`-100`であることをpreflightで確認した。
- Qwen3.5は32層のうちlinear attention 24層、full attention 8層。実ロード後にlanguage model側のattention / linear attention / MLP投影を含む248個のLinear層を検出し、PEFT適用後も248個と一致した。Vision encoder、embedding、LM headは除外。MPSのreference linear-attention kernelsは遅いため、rank 8 pilotで速度と学習曲線を記録する。

## 設定案

`config.yaml` に初期候補を記録した。BF16 LoRA、batch size 1、gradient checkpointingを使用する。pilot validationは固定seedの200件。公式processorで測ったtrain系列長p95=375から最大長512を選び、assistant回答を切らず、長い回答は除外して古い文脈を先に削る。全データではなく、held-out出力で安定性を確認してから学習量を増やす。

## 開始条件

1. 完了: `Qwen/Qwen3.5-4B` の公式post-trained Safetensors revisionを固定し、ローカル取得した。Base版とローカル4bit変換は使用しない。
2. 完了: Python環境とTransformers v5 / PEFT / TRLを用意し、隔離GPUプロセスからMPS/BF16を確認した。
3. 完了: 実モデルのtext側Linear層248個とPEFT適用件数248個をpreflightで記録した。
4. 完了: Qwen3.5 training template由来のassistant maskとcollator labelsをtoken単位で照合した。
5. 既定thinking prefixを使ったrank 8/16 pilotを実施したが、推論条件との不一致と反復出力が見つかったため選定から除外した。
6. `enable_thinking=false`を学習・推論双方に揃える修正を加え、TRLのassistant-only maskを再検証。rank 8 / 250例 / 1 epoch / LR 1e-5と未調整1,000例runは短文偏重のため不合格。
7. 低比重flag付き単発reactionを25%にするbalanced samplingを追加。rank 8 / 1,000例、rank 16 / 1,000例、rank 8 / 5,000例の修正版pilotを完了した。いずれもheld-out自動指標でbase超えを確認できず、rank 8 / 5,000例の出力中央値は18文字、char ROUGE-L-F1は0.0636（base 0.0912）。

## ライブラリ参照

[Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) / [Transformers Qwen3.5 docs](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5) / [TRL chat templates](https://huggingface.co/docs/trl/chat_templates)
