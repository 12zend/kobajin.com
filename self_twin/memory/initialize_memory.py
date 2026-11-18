#!/usr/bin/env python3
"""Create the local SQLite memory database from approved JSONL records only."""

import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "memory_database.sqlite"
JSONL_PATH = HERE / "personal_memory.jsonl"
SCHEMA_PATH = HERE / "schema.sql"


def main() -> None:
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        with JSONL_PATH.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("subject") != "self" or record.get("review_status") != "confirmed":
                    raise ValueError(f"Line {line_number} is not a confirmed self-memory")
                if float(record.get("confidence", 0)) < 0.8:
                    raise ValueError(f"Line {line_number} is below the memory confidence threshold")
                connection.execute(
                    """INSERT OR REPLACE INTO memories
                    (memory_id, content, category, observed_at, source_message_id, confidence,
                     is_current, could_be_stale, subject, review_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'self', 'confirmed')""",
                    (
                        record["memory_id"], record["content"], record["category"],
                        record.get("observed_at"), str(record["source_message_id"]),
                        float(record["confidence"]), int(bool(record.get("is_current", True))),
                        int(bool(record.get("could_be_stale", False))),
                    ),
                )
        connection.commit()
        count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        print(f"Initialized {DB_PATH} with {count} confirmed self memories")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
