# Merged model artifacts

元モデルを変更せず、新しいフォルダに出力する。

`qwen35_4b_self_twin_experimental_20261118/` は旧rank 8 / balanced 5,000例の実験マージ。`qwen35_4b_self_twin_experimental_luna_synthetic_v3_20261119/` はLuna作成合成データを70%含めたrank 8 v3の実験マージで、現在のHugging Face公開bundleの元モデル。v3は現行公開候補との同一条件評価で改善したためownerが実験版としての公開を明示的に許可した。一方、非適応baseには実会話指標で及ばず、品質ゲートは未通過のまま。各マージレシートにbase revision、LoRA hash、評価状態、公開承認を記録する。

`qwen35_4b_self_twin_kobajin_clean_v1_20261121/` と `qwen35_4b_self_twin_kobajin_clean_v2_20261121/` は、実会話データで学習したv1・v2のマージ。2026-11-21にownerが「学習した全バージョンをHFに上げ、サイトで選べるようにする」と決めたため、`training/merge_adapter.py --owner-release v1` / `--owner-release v2` でマージした。この方式は評価ゲートを使わず、レシートに `owner_release_version` を記録する。公開しても他バージョンやbaseより良いとは主張しない。評価結果は `training/training_report.md` にある。

owner公開方式以外の通常のマージは、引き続き評価ゲートを要求する。ownerが実験公開を明示した候補だけ、`training/merge_adapter.py --allow-unapproved-experimental` を使い、実験版と分かる別フォルダへ出力する。元のbase checkpointと各LoRA実験は保持する。
