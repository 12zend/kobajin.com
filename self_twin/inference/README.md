# ローカル推論

`inference.py` はローカルに保存したマージ済みモデルを使う薄い推論CLI。QwenのProcessor Chat Templateをそのまま使用する。外部サービスへプロンプトや記憶を送信しない。

過去の出来事、人物、プロジェクト、日時を尋ねる語が入力にある場合だけ、`memory/retrieve_memory.py` を呼び出す。検索結果は本人の確定記憶（confidence 0.8以上）に限る。現在は出典message_id付きの確認済み記憶3件をSQLiteに登録している。推論のChat Templateは学習と同じ`enable_thinking=false`を使う。

マージ済みモデルは評価ゲートを通過していないため、CLIの既定パスではまだ実行できない。評価合格後にマージし、MPSのBF16ロードとChat Templateを確認してから使う。

## サイト（WebLLM）との一致

`generation.py` のサンプリングはサイトのWebLLMと同じ条件に合わせている（2026-11-21から）。

- repetition penaltyは、その返事で生成したトークンだけに掛ける。Transformers標準の`repetition_penalty`はプロンプト全体（system prompt、記憶候補、相手の発言）にも掛かるが、WebLLMは生成したトークンしか数えない。
- `top_k`は使わない。Transformersは未設定の`top_k`を50で埋めるが、WebLLMにtop-kの打ち切りはない。

この変更より前に保存した評価結果は、上の2点がサイトと違う条件で出したもの。サイトとの差の測り方は `webllm/README.md` の「WebLLMとTransformersの一致」を参照。
