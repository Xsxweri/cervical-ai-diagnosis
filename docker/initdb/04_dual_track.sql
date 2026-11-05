\set ON_ERROR_STOP on

ALTER TABLE diagnosis_reports ADD COLUMN IF NOT EXISTS final_risk        TEXT;
ALTER TABLE diagnosis_reports ADD COLUMN IF NOT EXISTS declared_risk     TEXT;
ALTER TABLE diagnosis_reports ADD COLUMN IF NOT EXISTS dual_diverged     BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE diagnosis_reports ADD COLUMN IF NOT EXISTS agent_items_json  JSONB DEFAULT '[]'::jsonb;
ALTER TABLE diagnosis_reports ADD COLUMN IF NOT EXISTS agent_meta_json   JSONB DEFAULT '{}'::jsonb;

COMMENT ON COLUMN diagnosis_reports.final_risk IS
    '双轨合并后的最终等级；risk_level 保留代码轨原值，两者不同即分歧';
COMMENT ON COLUMN diagnosis_reports.declared_risk IS
    '从报告文本提取的 LLM 声明等级，用于统计分歧率';

CREATE INDEX IF NOT EXISTS idx_reports_diverged
    ON diagnosis_reports (created_at DESC) WHERE dual_diverged;

INSERT INTO schema_migrations (version) VALUES ('004_dual_track')
    ON CONFLICT (version) DO NOTHING;