# 学習レポート

## 状態

2026-11-21 v2の評価結果と、全バージョン公開への方針変更。

- 学習: 17,400 stepsで完了（約14時間）。最終validation loss `3.357`。adapter sha256 `7a7a7721888c94f50f6f9ca2fba6c65c7f9d286cf0665ea27e5ca53ce91887ba`。base revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- likeness評価（test 240件NLL、本番prompt）: ck2000 `3.846` → ck8000 `3.702` → ck12000 `3.660` → ck16000 `3.640` → 最終 `3.639`。v1最終 `3.617`、v3 `4.332`、base `4.826`。v1よりわずかに高いが、学習データの「うんこ」や【参加者】行を減らした分の差として小さい。最終adapterの生成60件は中央値9.5字（実返事9.5字）、5字以下33%、平均1.7行。「うんこ」・匿名化ラベルは0件。最終がもっとも良いため最終adapterを選んだ。
- 固定35件（本番prompt、温度0.72、top_p 1.0、repetition penalty 1.12、seed 42）: 記憶質問はRAGありで民謡「私は昔民謡を習っていました」、ナナチキ「僕はナナチキ大好き」と2/3で記憶を使った（v1 0/3、v3は曖昧な1/3）。TextBlurは「うん」で使えなかった。記憶質問のchar ROUGE-L-F1はv2 `0.323`、v3 `0.224`、v1 `0.030`。会話31件は `0.0952`（v1 `0.0745`、v3 `0.0999`、base `0.1213`）。【参加者】・【URL】の漏れは0件（v1 2件）、「うんこ」は3/35件（v1 5件）。中央値10字。v1と同じ相づち（「ohhh」「www」「えぐい」）が多く、「今から2時間かけて3D作るぞ！！」のような作り話も残る。民謡の答えの一人称が「私」だった。
- 記憶テスト（`evaluation/memory_use_eval.py`、test splitから関係する記憶29件と無関係な記憶40件）: 関係する記憶を使えた件数はbase 6/29、v3 1/29、v1 0/29、v2 27/29。無関係な記憶を誤って使った件数はbase 3/40、v3 0/40、v1 0/40、v2 6/40。v2の多くは記憶の発言をほぼそのまま繰り返す形で使っていた。無関係な記憶の混入（15%）はv1・v3より悪く、今後の課題。
- 方針変更: ownerが2026-11-21に「学習した全バージョンをHFに上げ、サイトで選べるようにする」と決めた（AGENTS.mdを更新）。評価は公開の可否ではなく、説明と初期値の判断に使う。v1とv2を`training/merge_adapter.py --owner-release`でマージし、q4f16_1に変換して`kobajin/kobajinAI`の`v1/`（commit [`8d6dab1`](https://huggingface.co/kobajin/kobajinAI/commit/8d6dab15195505c8c92c2d9ceb99f2551239f2d4)）と`v2/`（commit [`1f9287d`](https://huggingface.co/kobajin/kobajinAI/commit/1f9287d837eb311f5d0ab9262240ab2ed362d87c)）にアップロードした。v3はリポジトリ直下（`a2534a2`）のまま。サイトの初期値は`src/closest_model.json`に書いたバージョンで、ownerの判断でv2にした。v1・v3より良いと言える点は記憶の利用と口調で、無関係な記憶の混入と作り話は残っている。

2026-11-20 v1の問題を直すための再学習v2（`adapter/experiment-r8-kobajin-clean-v2-memory/`）を開始した。ベースは同じrevision。データは`training/build_kobajin_train_v2.py`でv1から作った`kobajin_clean_v2`（train 17,405行）。

- 記憶の不使用への対策: サイトと同じ形式（本番system prompt＋`getMemoryContext()`の記憶ブロック）の行を追加した。記憶候補には、実返事に含まれる本人の発言そのもの（144件）を、確認済み記録と同じ「本人が「…」と発言」という形で引用する。各発言から、無関係な記憶0〜2件を混ぜた2行を作って288行。さらに、会話と無関係な記憶だけが入っていて実返事が記憶を使わない行を400行入れた。評価で使う3件の記憶（民謡・TextBlur・ナナチキ）とその出典メッセージは使っていない。疑問文と6字未満の発言は記憶候補から外した。
- 「うんこ」の多用への対策: 「うんこ」を含むtargetを1/3に間引いた（80行を除外）。
- 匿名化ラベル漏れへの対策: 「【参加者】」を含むtarget 641行を除外し、targetから「【URL】」を取り除いた。validationにも同じ除外を適用した（2,705行）。
- prompt: 本番promptだけの行を15%（2,557行）混ぜた。記憶行も本番promptなので、本番promptの行は全体の約20%。max sequence lengthは768（本番prompt行が512を超えるため）。
- 学習設定はv1と同じ。rank 8、LR 5e-5、cosine、1 epoch、17,400 steps、2,000 stepごとにcheckpointを保存。smoke test（24例）は正常に完了した。
- 評価: 終了後に全checkpointのlikeness評価、固定35件のbenchmark、`evaluation/memory_use_eval.py`を自動で実行する。memory_use_evalはtest split（学習に未使用）から作った記憶の使用テストで、関係する記憶29件と無関係な記憶40件を使い、base・v3・v1・v2を同じ条件で比べる。

2026-11-20 本人らしさ重視のbase再学習（`adapter/experiment-r8-kobajin-clean-v1-lr5e5/`）の準備と開始。学習前に次を確認・変更した。

- 評価: `evaluation/kobajin_likeness.py`を追加。時系列test（学習に未使用）240件で本人の実際の返事のtoken NLLと、60件をサイトと同じ設定で生成した文体プロファイル（長さ、敬語、アシスタント口調、合成テンプレ句、placeholder、反復）を測る。基準値はbase NLL `4.826`、公開中v3 `4.332`、full_v1 `3.691`（本番system prompt）。本人の実返事は中央値9.5字・5字以下32%（test全体は14字・27%）。v3は合成テンプレ句（「小さく試」など）が5%出て、本人の実データにない口癖を持ち込んでいた。char-bigram JSDは長文ほど低く出るため判断材料にしない。
- system prompt: 学習時の短いpromptと本番promptのNLL差はfull_v1で`0.016`と小さかった。本番promptで学習すると系列長がほぼ倍（+284 token）になり学習時間も約2倍になるため、短いpromptのまま学習する。
- 短文化の原因: full_v1は温度1.0・反復penaltyなしでも5字以下が47%で、学習データ（16%）や実返事（27%）より多かった。デコード設定ではなくモデルが汎用の相づちに逃げる未学習側の症状と判断し、LRを`1e-5`から`5e-5`（cosine、warmup 3%）へ上げた。デコードは温度1.0・反復penalty 1.12が本人の長さ分布に最も近かった（中央値13字・5字以下30%）。サイトの温度0.72は短文を増やす。
- データ: `training/build_kobajin_train.py`で実会話だけの派生split `kobajin_clean_v1`を作成。合成例は混ぜない（テンプレ句の持ち込みを避けるため）。Discordの`(edited)`を除去、placeholderだけの返事18件と`@通話募集`だけの返事144件を除外、同一返事は20件まで（950件除外）、低比重reactionは50%。train 17,838 / validation 2,770（元splitは不変）。
- 学習: rank 8、1 epoch、17,830 steps、2,000 stepごとにvalidation 300件で評価してcheckpointを全保存（lossだけでなくlikeness評価でcheckpointを選ぶため）。`--resume`で最新checkpointから再開でき、smoke testで再開後のlossが一致することを確認。`caffeinate`で実行し、約2.25秒/step、約11〜12時間の見込み。
- step 2000（11%）の中間確認: validation loss `3.544`、grad normは減少傾向でNaNなし。test 240件のNLLは`3.819`（base `4.826`、v3 `4.332`、full_v1最終`3.691`）。サイト設定（温度0.72）で30件生成すると、中央値16.5字、5字以下27%、平均2.23行で、実返事の分布（14字・27%・2.3行）に近づいた。敬語、アシスタント口調、合成テンプレ句は0件。ただし「ほうほうほう」のような反復が1件、「親は最近仕事忙しい」など本人の事実として確認できない話が出たため、最終選定では反復と作り話を重点的に見る。
- 学習終了後に`evaluation/eval_checkpoints.sh`が全checkpointと最終adapterを同じtest行で自動評価する（出力`evaluation/kobajin_likeness-clean-v1-all.json`）。
- 途中経過: 12:24ごろにMacが予期せず再起動し、step 4017で停止した。checkpoint-4000から`--resume`で再開し、21:50に17,830 stepsまで完了した（学習時間合計約9.3時間）。最終validation loss `3.2145`。adapter sha256 `75feda45165454f6cfca0670f0d641f41926da81fb9edc26dfd534cb4d71b4ee`。base revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- likeness評価（test 240件NLL、本番prompt）: ck2000 `3.819` → ck4000 `3.729` → ck8000 `3.661` → ck12000 `3.650` → ck14000 `3.622` → ck16000 `3.616` → 最終 `3.617`。比較はbase `4.826`、v3 `4.332`、full_v1 `3.691`。最終adapterの生成60件は中央値7.5字、5字以下35%、平均1.65行（実返事は9.5字・32%・2.08行）。敬語・アシスタント口調・合成テンプレ句は0件。checkpoint間の文体差は60件のsampling誤差の範囲で、NLLは16000以降ほぼ横ばい。
- 固定35件（本番system prompt、温度0.72、top_p 1.0、repetition penalty 1.12、seed 42）での最終adapter: 会話31件のchar ROUGE-L-F1は `0.0745`（v3 `0.0999`、base `0.1213`）。中央値11字、5字以下10/35、100字超0、12字以上の反復span 0件。v3はほぼ全件に「小さく試そ」「全部を一回やるより」などの合成テンプレ句が入り、本人の口調から外れていた。新adapterでは消え、「あーね」「えぐい」「おおおおお」のような実ログ由来の言い方になった。char ROUGEは長い返答ほど高く出やすく、短い返答が多い新adapterには不利なので、補助指標として扱う。
- 残った問題（material regressionと判断）: (1) 記憶の利用。RAGで正しい記憶を3/3取得しているのに、使えたのは0/3（v3は1/3）。例として、民謡を習っていた件に「ピアノよ！！！」、ナナチキの件に「カステラ」と答えた。(2) 「うんこ」の多用。実返事では0.78%だが、35件中5件（14%）、likeness生成でも60件中2件に出た。(3) 本人の事実として確認できない話を作る（「18時間くらい」「2時間くらいやってる」など）。(4) 匿名化ラベル「【参加者】」の漏れが2件。ck4000（記憶1/3、うんこ1件、5字以下11/35）とck8000（記憶0/3、うんこ2件、5字以下17/35）も同条件で測ったが、どれも同時には解決しない。合成held-out 20件のchar ROUGE-L-F1は `0.0632`（v3 `0.2436`）。ただしこれはLunaが書いた合成応答への追従を測る指標で、今回は合成データを除いたので下がるのは想定どおり。
- 判断: 口調の本人らしさと実返事の予測（NLL）はv3より明確に良い。一方、記憶の利用と作り話は後退したため、AGENTS.mdの条件（material regressionなし）を満たさない。merge・変換・Hugging Faceへのアップロードは行っていない。出力は`evaluation/kobajin_likeness-clean-v1-all.json`、`evaluation/benchmark-r8-kobajin-clean-v1-{final,ck4000,ck8000}-sampled.json`、`evaluation/benchmark-synthetic-heldout-r8-kobajin-clean-v1-final-sampled.json`。

2026-11-20のrank 8 full run（`adapter/experiment-r8-authored-synthetic-2000-full-v1/`、adapter sha256先頭`087f37bc6747c924`）: 全train 20,145例（通常13,882・低比重reaction 4,468・新合成1,800、合成比率約9%）、1 epoch、LR `1e-5`、20,145 steps、約12.5時間。train loss `3.480`、validation loss `3.301`（2,991行。v3の312行とは集合が違うため直接比較しない）。lossは単調に下がりNaNなし。adapterは248 moduleでbaseへ正常に読み込め、生成も停止した。評価はv3と同じprompt・system prompt（sha256 `4bc750d8…`）・sampling（temperature 0.72、top_p 1.0、repetition penalty 1.12、seed 42 / 73）で、baseの出力は前回runを再利用した。実会話31件の会話返信char ROUGE-L-F1はbase `0.1213`、公開中v3 `0.0999`、full `0.0841`。返答中央値はv3 `34`文字に対してfull `9`文字、5文字以下は`3/35`から`15/35`に増えた。既知記憶3件はRAGで3件とも出典を取得したが、回答は「思い出せない」「うん」「バナナ」で事実を返さなかった（v3はナナチキを正答）。未知記憶への回答は不適切な短文だった。合成held-out 20件はbase `0.178`、v3 `0.244`、full `0.072`、中央値`10`文字、5文字以下`9/20`、反復1件。具体的な相談にも「あーね」「ほう」など相づちだけを返す傾向が強く、合成比率が低い全量学習で短いDiscord reactionへ戻ったと判断する。v3より劣るためmerge・変換・HF公開は行わない。

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
