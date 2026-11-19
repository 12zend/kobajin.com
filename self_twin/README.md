# Self-twin project

このディレクトリは、Discord履歴から本人の会話スタイルを学習するローカル専用プロジェクト。

## 現在の状態

- `analysis/`: user_id監査、会話スタイルの集計、暫定プロフィールを作成済み。
- `dataset/`: 3アカウントを統合し、連続投稿をまとめ、強い話題転換59か所を分割した時系列SFTデータを生成済み。
- `dataset/synthetic_explanations_*.jsonl`: Discordで観察した話題と文体の手掛かりをもとに作った合成例2,000件（200シナリオ×10 intent、train 1,800・validation 200）。Scratch、ソフトウェア、3D、Web、映像、LLM、共同制作の具体的な質問を扱う。入力と回答は全件ユニークで、validationはシナリオ単位で分離。Discord本文や本人の記憶・好み・信条は含めない。
- `memory/`: 本人IDの明示的な過去経験・プロジェクト・好みから、出典付き確定記憶3件を作成。必要な質問だけローカル検索する。
- `training/`: official post-trained Qwen3.5-4Bを固定revisionで取得。MPS/BF16、248個の言語側LoRA target、TRL assistant-only maskのtoken単位検証を実施。rank 8 / 1,000例pilot v3（合成70%）はtrain loss 2.955、validation loss 3.238で完了。v3は現在のHF公開候補rank8 / 5,000例より、実会話の返答中央値とchar ROUGE-L-F1、合成held-outの追従が改善したため、owner指示で実験版として公開した。baseとの品質ゲートは未通過。
- `evaluation/`: 時系列testから35件、held-out合成シナリオ20件を評価。現在の本番system promptと同じファイルをWebLLMとベンチマークで共有し、同じsampling条件でbase、現行HF候補、v2、v3を比較する。現行HF候補→v3は実会話会話返信F1 `0.049`→`0.100`、返答中央値5→34文字、5文字以下18→3件。合成held-out F1は`0.075`→`0.244`。ただし実会話ではbase `0.121`に届かない。RAG記憶と会話スタイルは別々に評価。
- `inference/`: ローカル推論と必要時のみの記憶検索をつなぐCLI案。未実行。
- `inference/system_prompt.txt`: WebLLM本番とheld-out評価で共通利用。質問・相談は具体的な返答にしつつ、相づちは短く保ち、本人の事実を作らない。
- `adapter/` と `merged/`: baseと分離して実験別に保存する。旧rank 8 / balanced 5,000例とv3を別フォルダへ保存。v3はbase品質ゲート未通過だが、現行公開候補より改善したためowner指示で実験版としてHF公開し、merge receiptに評価状態を記録。
- `webllm/`: WebLLM向けの静的チャットサイト用に、モデルとHugging Faceへアップロードするファイルを案内する。

## 再生成

```bash
self_twin/.venv/bin/python self_twin/training/prepare_dataset.py \
  --input /Users/kobajin/Downloads/chat_messages.json \
  --output self_twin/dataset \
  --model-id Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --cache-dir self_twin/hf_home/hub

python3 self_twin/evaluation/build_evaluation_prompts.py
python3 self_twin/memory/import_confirmed_memories.py
```

前処理本体はPython標準ライブラリで動き、元のJSONやベースモデルを変更しない。再生成時に正確なチャット形式token数も記録するには、下記コマンドのようにproject venv内のTransformersとローカルに取得済みの固定revisionを使う。

## 学習前ゲートの結果

1. 完了: `Qwen/Qwen3.5-4B` post-trained Safetensors revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` を固定。
2. 完了: MPS/BF16でモデルをロードし、Qwen3.5のAttention / Gated DeltaNet / MLP各投影を含む248個のtext Linear層だけをLoRA対象にした。
3. 完了: TRLのQwen3.5 training templateで、保存されたlabelsとassistant maskが一致し、promptがloss対象外であることを実測。
4. EOS停止修正後の4候補を再評価。rank 8 / 5,000例は会話31件char ROUGE-L-F1が0.0577（base 0.0901）、5ケースで反復、24/35件が5文字以下。停止修正版の主観評価を待ち、追加学習・マージは保留。

学習スクリプトは公式repo ID `Qwen/Qwen3.5-4B` のみを読み、revisionを必須にし、`local_files_only=True` で実行する。rank 8のpilot例:

```bash
python3.13 self_twin/training/train.py \
  --revision <immutable-official-commit> \
  --rank 8 \
  --output-dir self_twin/adapter/experiment-r8-topic-aware
```

既定では1,000件だけ学習する。修正版データで両rankを比較し、validationとheld-out出力を評価する。全学習splitを使う場合はpilot比較後に `--pilot-examples 0` を明示する。

説明・相談への返答を補うpilotでは、合成セットを別ファイルで指定する。新しいセットは200件の汎用シナリオを10種類の入力意図へ展開し、同じシナリオがtrainとvalidationへまたがらない。本人の過去や好みを例文に作らず、短い要約から技術説明まで回答の深さを分けた。生成はスクリプトで再現できる。

```bash
python3 self_twin/training/build_synthetic_dataset.py
self_twin/.venv/bin/python self_twin/training/train.py \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --rank 8 --pilot-examples 1000 --epochs 1 --learning-rate 0.00001 \
  --low-weight-sample-rate 0.5 --synthetic-share 0.70 \
  --synthetic-train self_twin/dataset/synthetic_explanations_train.jsonl \
  --synthetic-validation self_twin/dataset/synthetic_explanations_validation.jsonl \
  --output-dir self_twin/adapter/experiment-r8-authored-synthetic-2000-v1
```

v3は合成例を70%含める。評価promptは本番system promptを使う。v3はv2より返答が長くなり、合成質問への追従も上がったが、実会話では具体性・関連性のずれが残り、baseを上回らなかった。ownerの明示承認で実験版として公開済みだが、品質ゲートは未通過のまま。今後の通常の昇格は品質ゲート通過後に行う。

チャット履歴はローカル処理し、外部サービスへ送っていない。現在のこの環境ではTransformers v5 / PEFT / TRLのMPS経路を使っている。Unslothはインストール・学習経路に使っていないため、Qwen3.5対応のUnslothガイドと必要VRAM目安は参考条件として記録し、Apple Siliconでの互換性がないという意味ではない。
