CREATE TABLE runtime_version (version INTEGER NOT NULL);
INSERT INTO runtime_version VALUES (1);
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    last_admitted INTEGER NOT NULL DEFAULT 0,
    agent_count INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE agents (
    id TEXT PRIMARY KEY,
    root_id TEXT NOT NULL REFERENCES runs(id),
    parent_id TEXT REFERENCES agents(id),
    spawn_key TEXT,
    parent_cutoff INTEGER,
    fresh INTEGER NOT NULL DEFAULT 0,
    compaction INTEGER NOT NULL DEFAULT 0,
    subscription_event INTEGER NOT NULL DEFAULT 0,
    closed INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'waiting'
        CHECK(state IN ('queued','running','waiting','paused','completed','failed','cancelled')),
    generation INTEGER NOT NULL DEFAULT 0,
    pause_requested INTEGER NOT NULL DEFAULT 0,
    lease_id TEXT,
    last_admitted INTEGER NOT NULL DEFAULT 0,
    checkpoint TEXT NOT NULL DEFAULT '{}',
    metadata TEXT,
    UNIQUE(parent_id, spawn_key)
);
CREATE INDEX agents_root ON agents(root_id, id);
CREATE INDEX agents_parent ON agents(parent_id);
CREATE INDEX agents_ready ON agents(state, root_id, last_admitted, id);
CREATE TABLE inbox (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    message_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    lease_id TEXT,
    processed INTEGER NOT NULL DEFAULT 0,
    UNIQUE(agent_id, message_id)
);
CREATE INDEX inbox_pending ON inbox(agent_id, processed, sequence);
CREATE TABLE history (
    agent_id TEXT NOT NULL REFERENCES agents(id),
    position INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(agent_id, position),
    UNIQUE(agent_id, message_id)
);
CREATE TABLE processed_inputs (
    agent_id TEXT NOT NULL REFERENCES agents(id),
    message_id TEXT NOT NULL,
    PRIMARY KEY(agent_id, message_id)
);
CREATE TABLE summaries (
    agent_id TEXT NOT NULL REFERENCES agents(id),
    cutoff INTEGER NOT NULL,
    summary TEXT NOT NULL,
    PRIMARY KEY(agent_id, cutoff)
);
CREATE TABLE subscriptions (
    agent_id TEXT NOT NULL REFERENCES agents(id),
    call_id TEXT NOT NULL,
    PRIMARY KEY(agent_id, call_id)
);
CREATE TABLE choices (
    agent_id TEXT NOT NULL REFERENCES agents(id),
    choice_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(agent_id, choice_id)
);
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    generation INTEGER NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('queued','dispatched','completed','uncertain','cancelled')),
    payload TEXT NOT NULL,
    result TEXT
);
CREATE INDEX jobs_agent ON jobs(agent_id, state);
