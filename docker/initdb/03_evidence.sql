\set ON_ERROR_STOP on

ALTER TABLE diagnosis_reports ADD COLUMN IF NOT EXISTS evidence_json    JSONB;
ALTER TABLE diagnosis_reports ADD COLUMN IF NOT EXISTS validator_issues JSONB;
ALTER TABLE diagnosis_reports ADD COLUMN IF NOT EXISTS evidence_sufficiency REAL;

ALTER TABLE diagnosis_reports
    ALTER COLUMN validator_issues SET DEFAULT '[]'::jsonb,
    ALTER COLUMN evidence_json    SET DEFAULT '[]'::jsonb;

COMMENT ON COLUMN diagnosis_reports.evidence_json IS
    '证据账本快照：每条带稳定 ID，报告引用可回放';
COMMENT ON COLUMN diagnosis_reports.validator_issues IS
    '报告校验问题列表，用于统计 blocker 率';
COMMENT ON COLUMN diagnosis_reports.max_uncertainty IS
    '模型测量值（P90 归一化熵）。注意：state 层的同名键可能被历史逻辑改写，'
    '本列始终取 cv_result 原值，是权威来源';

UPDATE diagnosis_reports
   SET validator_issues = '[]'::jsonb
 WHERE validator_issues IS NULL;
UPDATE diagnosis_reports
   SET evidence_json = '[]'::jsonb
 WHERE evidence_json IS NULL;

INSERT INTO schema_migrations (version) VALUES ('003_evidence')
    ON CONFLICT (version) DO NOTHING;
