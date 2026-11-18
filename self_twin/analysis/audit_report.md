# データ・実行環境監査

監査日: 2026-11-18 (Asia/Tokyo)

## データ監査

対象: `/Users/kobajin/Downloads/chat_messages.json`  
SHA-256: `06de0f0ef5408cf034e93e044b6415949639af380e7168ac3aa800419de7d938`  
サイズ: 50,251,044 bytes  
チャンネル: `Japanese Intro Makers Organization / VC / 聞き専`

| 本人アカウント | user_id | 期待投稿数 | 実数 | 差 |
|---|---:|---:|---:|---:|
| zend3d / 12zend | `1120529566138634275` | 26,806 | 26,806 | 0 |
| kobajin_ / ルイボスティー | `1206039595129639002` | 20,313 | 20,313 | 0 |
| 12zend / triangle.area | `1262554507233984576` | 5,567 | 5,567 | 0 |
| **合計** | 3 IDを同一人物として統合 | **52,686** | **52,686** | **0** |

投稿者の判定は `author.user_id` のみを使った。表示名が `Deleted User` の投稿21,130件は本人IDではないため、本人の学習ターゲットに含めていない。全205,491件のうち、本人以外の投稿は会話文脈としてのみ利用した。

投稿IDは配列順に単調増加している。Discord Snowflake時刻を復元し、表示時刻を解析できた134,568件すべてで差が90秒以内（実測は最大60秒未満）だった。この検証後、欠損時刻をSnowflakeから補った。本人投稿52,686件のうち表示時刻の欠損は22,819件。会話の元配列順は変更していない。

`reply_to` は14,669件にあり、返信先の本文・投稿者・message_idを含む辞書形式だった。13,390件はエクスポート内の投稿IDに解決できた。本人の投稿では1,791件が返信。返信先は同じ6時間無投稿で区切ったセッション内なら会話入力の先頭に置き、別セッションをまたぐ19サンプルでは分割間の情報漏洩を避けて返信先を省いた。返信先以外の直近12件も文脈に使った。

## 投稿の統合とSFTデータ

本人の連続投稿ペアは**23,861組**で、指定された件数と一致した。これは配列上で本人投稿が隣接し、間に他人の投稿がないペア数。10分以内かつ間に他者投稿がない本人投稿をまとめると29,267バースト（単独投稿を含む）になった。加えて「話変わる」「別の話」「ところで」「そういえば」など明示的な話題転換を59か所で検出し、バーストを分割した。短い時間内の複数投稿は改行で保持した。明示的な話題転換語がないまま意味的に切り替わるケースを完全検出する分類器はなく、時間・投稿者・明確な転換表現を使った保守的な分割である。

| 項目 | 件数 |
|---|---:|
| 元の本人投稿 | 52,686 |
| 生成SFTサンプル | 28,451 |
| 収録された本人投稿 | 50,790 |
| 除外された本人投稿 | 1,896 |
| セッション | 594 |
| train / validation / test | 22,818 / 2,791 / 2,842 |

セッションは6時間の無投稿で分け、セッションをまたがない順序で80.2% / 9.8% / 10.0%に分割した。時期はtrainが2024-04-14〜2026-09-20 JST、validationが2026-09-20〜2026-10-15 JST、testが2026-10-15〜2026-11-17 JSTで、時系列の重複はない。

| 除外理由 | 件数 |
|---|---:|
| 空白投稿 | 1,005 |
| URLだけの投稿 | 715 |
| コマンド形式 | 131 |
| 10分内の長文完全重複 | 36 |
| 電話番号らしい文字列 | 2 |
| メールアドレス | 5 |
| 数字を伴う連絡先・住所表現 | 2 |

短い反応も文脈がある限り残した。サンプル分類は短い反応11,009、会話13,191、質問・促し2,398、冗談・リアクション1,275、技術・制作369、長文説明・体験談209。短い反応は38.7%で、全体の会話スタイルを保つため残した一方、将来の学習ではカテゴリ別評価を見て過剰な短文生成が起きないか確認する。

学習JSONLには会話ごとに匿名の「参加者A/B…」を割り当て、投稿者名、表示名、user_idを入れていない。既知の投稿者名、Discordメンション・ID、URL、電話番号、メールアドレスを文脈から匿名化した。連絡先や認証情報らしい本人投稿はターゲットから除外した。エクスポートにbot属性がないため、bot判定はコマンド形式などのヒューリスティックに限られる。また、投稿本文中に直接書かれた未登録の第三者名や秘密を完全に識別する仕組みはなく、生成データは追加レビューが必要。

### トークン数

公式チェックポイントのrevision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` のProcessorとChat Templateで、各SFT会話をレンダリングした正確な系列長を計算した。train / validation / testの系列トークン数（会話template込み）は6,996,616 / 858,124 / 876,157。系列長のp50 / p90 / p95 / 最大はtrainが303 / 354 / 375 / 10,681、validationが307 / 366 / 390 / 666、testが303 / 370 / 396 / 1,210。長い外れ値があるため学習時はassistant回答を切り詰めず、古い文脈から調整する。

## 実行環境

| 項目 | 実測 |
|---|---|
| OS | macOS 27.0.1 (Darwin 27.0.0, arm64) |
| CPU / GPU | Apple M6、CPU 12コア、内蔵GPU 12コア、Metal対応 |
| RAM / GPUメモリ | 統合メモリ32 GB。独立したGPU VRAMはない |
| 空きストレージ | 458 GiB |
| 既定Python | 3.15.0 |
| 利用可能なPython | 3.13.18（この監査・生成で使用） |
| PyTorch / CUDA | project venv: PyTorch 2.15.0、CUDAなし |
| Transformers / PEFT / TRL | project venv: Transformers 5.23.0 / PEFT 0.21.3 / TRL 1.18.1 |
| その他 | datasets 5.2.0 / accelerate 1.16.0 / tokenizers 0.23.3 / huggingface_hub 2.4.0 |
| Unsloth / MLX | 未導入・未使用。現在のpilotは実測したTransformers / PEFT / TRLのMPS経路 |
| MPS | GPU device access付きプロセスでは利用可能。BF16行列演算も動作確認済み |

通常権限の隔離プロセスからはMetal GPUが見えず、MPSは利用不可だった。ローカルGPUアクセスを許可した実行ではMPSが見つかり、BF16行列演算が成功した。学習プロセスはこのGPUアクセスが有効な状態で実行する。Unslothはインストール・使用していないため、このMacに非対応とは結論づけない。

## ベースモデルの確認

環境に既存だったモデルは次のMLX変換モデル。

- ローカルパス: `/Users/kobajin/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- ローカル変換リポジトリrevision: `32f3e8ecf65426fc3306969496342d504bfa13f3`
- 形式: MLX SafeTensors、4-bit affine、group size 64、約3.03 GB
- `config.json`: `Qwen3_5ForConditionalGeneration`、32層（linear attention 24層、full attention 8層）
- `README.md` の `base_model`: `Qwen/Qwen3.5-4B`
- Tokenizer SHA-256: `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`
- Chat template SHA-256: `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`
- Chat template内に `{% generation %}` / `{% endgeneration %}` はない。

これは通常版 `Qwen/Qwen3.5-4B` 由来とされるpost-trainedモデルだが、元の公式revisionは変換メタデータに固定されていない。Tokenizer hashも公式モデルと異なるため、学習には流用していない。

学習用として公式post-trainedモデルを別途取得し、以下を確認した。

- repo: `Qwen/Qwen3.5-4B`（`Qwen/Qwen3.5-4B-Base`ではない）
- immutable revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- 保存先: `self_twin/hf_home/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- 形式: BF16対応の未量子化Transformers Safetensors、2 shards、合計9,319,828,096 bytes
- `config.json`: `qwen3_5` / `Qwen3_5ForConditionalGeneration`、32層（linear attention 24層、full attention 8層）
- config SHA-256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`
- Tokenizer SHA-256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- Chat template SHA-256: `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`
- Chat templateに `{% generation %}` / `{% endgeneration %}` はない。

取得時のrevision、サイズ、hashは`training/base_checkpoint.json`に記録した。公式Tokenizer hashは既存MLX変換版と異なり、データのトークン計数と学習には公式チェックポイントを使用した。

Ollamaのローカルmanifestには `Qwen3.8-27B-Heretic-JP-Roleplay:q4_k_m` が見つかったが、Ollama daemonへの接続が許可されず `ollama list` で実行中モデルは確認できなかった。これは指定の4Bベースとは別物。現在アクティブなアプリ上のモデルまでは確定していない。

## 学習準備の判断

モデルは取得済みで、Transformers v5 / PEFT / TRLをプロジェクト専用venvに導入済み。実モデルのLinear層248個とLoRA適用248個が一致し、MPS/BF16を確認した。TRLのQwen3.5用templateでcollator labelsとassistant generation maskが一致し、system/user promptにlossがかからないこともtoken単位で確認した。最初のrank 8/16 pilotでは推論のthinking設定と学習が不一致だったため選定対象から外した。現在のpilotは学習と推論で`enable_thinking=false`をそろえている。UnslothのQwen3.5ガイドはTransformers v5を要求し、4B BF16 LoRAのVRAM目安を約10 GBとし、4bit QLoRAを推奨していない。本実験環境はApple M6の統合メモリ/MPSなので、VRAM値は直接適用せず、Unsloth経路を使っていない理由を互換性不足と取り違えない。MPSでは`causal_conv1d`と`flash-linear-attention`がないため、線形attentionはPyTorch reference実装へfallbackする。小規模pilotのheld-out応答に短文偏重や反復が残るため、データ分布と生成品質を見直しながら次の実験を進めている。

ローカルChat Template自体にはassistant生成領域のマーカーがないが、TRL 1.18.1が用意するQwen3.5用training templateに置き換わる。collatorのlabelsとtemplateのassistant maskをtoken単位で照合し、プロンプトはすべて`-100`、回答領域のみloss対象となることを実測した。さらに学習・推論双方で`enable_thinking=false`を渡し、format mismatchを防いでいる。

Qwen3.5はGated DeltaNetとfull Attentionのハイブリッド構造。実モデルのtext側からAttention / Linear Attention / MLP投影を列挙し、248層すべてにLoRAが適用されたことを確認した。Vision tower・embedding・LM headは対象外。

## 参照した公式情報

- [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) はTransformers形式のpost-trainedモデル。
- [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) は別のpre-trained-onlyモデル。
- [ローカルにあるMLX 4bit変換モデルのmodel card](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) は元モデルを `Qwen/Qwen3.5-4B` と記載。
- [Qwen3.5 Transformers model documentation](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5) は3:1のGated DeltaNet / Gated Attention構造を説明。
- [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) はTransformers v5とBF16 LoRA要件を記載し、4bit QLoRAを推奨していない。
- [TRL chat template guide](https://huggingface.co/docs/trl/chat_templates) はassistant-only loss用の生成mask要件とQwen3.5向け学習templateを説明。
- [PEFT model reference](https://huggingface.co/docs/peft/en/package_reference/peft_model) は`merge_and_unload()`を説明。
