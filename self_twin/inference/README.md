# ローカル推論

`inference.py` はローカルに保存したマージ済みモデルを使う薄い推論CLI。QwenのProcessor Chat Templateをそのまま使用する。外部サービスへプロンプトや記憶を送信しない。

過去の出来事、人物、プロジェクト、日時を尋ねる語が入力にある場合だけ、`memory/retrieve_memory.py` を呼び出す。検索結果は本人の確定記憶（confidence 0.8以上）に限る。現在は出典message_id付きの確認済み記憶3件をSQLiteに登録している。推論のChat Templateは学習と同じ`enable_thinking=false`を使う。

マージ済みモデルは評価ゲートを通過していないため、CLIの既定パスではまだ実行できない。評価合格後にマージし、MPSのBF16ロードとChat Templateを確認してから使う。
