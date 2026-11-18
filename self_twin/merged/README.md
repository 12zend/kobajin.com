# Merged model artifacts

元モデルを変更せず、新しいフォルダに出力する。

`qwen35_4b_self_twin_experimental_20261118/` は、ownerの依頼で作成した実験マージ。評価記録では品質ゲート未通過のため、合格済みまたは品質改善済みとして扱わない。マージレシートに候補、評価状態、ベースrevision、LoRAハッシュを記録する。

通常のマージは引き続き評価ゲートを要求する。未承認候補を試す場合だけ、`training/merge_adapter.py --allow-unapproved-experimental` を明示する。元のbase checkpointと各LoRA実験は保持する。
