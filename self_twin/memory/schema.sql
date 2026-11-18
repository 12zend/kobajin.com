CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    observed_at TEXT,
    source_message_id TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    is_current INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    could_be_stale INTEGER NOT NULL DEFAULT 0 CHECK (could_be_stale IN (0, 1)),
    subject TEXT NOT NULL DEFAULT 'self' CHECK (subject = 'self'),
    review_status TEXT NOT NULL DEFAULT 'confirmed' CHECK (review_status IN ('candidate', 'confirmed')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS memories_category_idx ON memories(category);
CREATE INDEX IF NOT EXISTS memories_source_idx ON memories(source_message_id);
CREATE INDEX IF NOT EXISTS memories_current_idx ON memories(is_current, review_status);
