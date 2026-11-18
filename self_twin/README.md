# Self-twin project

このディレクトリは、Discord履歴から本人の会話スタイルを学習するローカル専用プロジェクト。

## 現在の状態

- `analysis/`: user_id監査、会話スタイルの集計、暫定プロフィールを作成済み。
- `dataset/`: 3アカウントを統合し、連続投稿をまとめ、強い話題転換59か所を分割した時系列SFTデータを生成済み。
- `memory/`: 本人IDの明示的な過去経験・プロジェクト・好みから、出典付き確定記憶3件を作成。必要な質問だけローカル検索する。
- `training/`: official post-trained Qwen3.5-4Bを固定revisionで取得。MPS/BF16、248個の言語側LoRA target、TRL assistant-only maskのtoken単位検証を実施。rank 8 / 1,000例、rank 16 / 1,000例、rank 8 / 5,000例、短い単発反応10%のrank 8 / 1,000例pilotを完了したが、会話のみのheld-out自動指標でbaseを上回る品質は確認できていない。
- `evaluation/`: 時系列testから30件の主観評価promptを抽出し、意見訂正1件、記憶想起3件、未確認記憶1件を追加。EOS停止修正後に4候補を再評価済み。古いrank 8 / 5,000例レビューはEOS未指定の出力に対する暫定採点。修正後の会話出力レビューを別シートに用意。RAG記憶と会話スタイルは別々に評価。
- `inference/`: ローカル推論と必要時のみの記憶検索をつなぐCLI案。未実行。
- `adapter/` と `merged/`: baseと分離して実験別に保存する。2026-11-18にowner依頼でrank 8 / balanced 5,000例を別フォルダへ実験マージした。評価decisionは品質ゲート未通過のままで、結果は承認済み・改善済みモデルではない。マージレシートを一緒に保存。
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

チャット履歴はローカル処理し、外部サービスへ送っていない。現在のこの環境ではTransformers v5 / PEFT / TRLのMPS経路を使っている。Unslothはインストール・学習経路に使っていないため、Qwen3.5対応のUnslothガイドと必要VRAM目安は参考条件として記録し、Apple Siliconでの互換性がないという意味ではない。
