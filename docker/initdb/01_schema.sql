\set ON_ERROR_STOP on

-- ------------------------------------------------------------
-- 账号
-- ------------------------------------------------------------ 
CREATE TABLE IF NOT EXISTS users (
    id              BIGSERIAL PRIMARY KEY,
    username        TEXT        NOT NULL UNIQUE,
    password_hash   TEXT        NOT NULL,          -- Werkzeug scrypt 格式，与旧 JSON 兼容
    name            TEXT,
    department      TEXT,
    hospital        TEXT,
    role            TEXT        NOT NULL DEFAULT 'doctor'
                                CHECK (role IN ('doctor', 'admin')),
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    failed_attempts INTEGER     NOT NULL DEFAULT 0,
    locked_until    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at   TIMESTAMPTZ
);

COMMENT ON COLUMN users.password_hash IS 'Werkzeug scrypt 哈希，不存明文';
COMMENT ON COLUMN users.locked_until  IS '连续失败达阈值后的解锁时间';

-- ------------------------------------------------------------
-- 诊断报告（会话冷存储）
--
-- 与 Redis 分工：Redis 存进行中会话（TTL 1 天），
-- 本表存已完成报告（永久，归档后才删）。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diagnosis_reports (
    session_id       TEXT        PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- 归属：Redis 的 session_owner 过期后，这里是权威来源
    operator_id      TEXT        NOT NULL,
    patient_ref      TEXT,                          -- 脱敏病例号，可空

    -- 判定结果
    label_space      TEXT        NOT NULL DEFAULT 'sipakmed',
    risk_level       TEXT,
    risk_scope       TEXT,
    review_required  BOOLEAN     NOT NULL DEFAULT FALSE,

    -- 冲突分项：区分「分级模型判非异常」与「双空间严重度不一致」
    conflict_count          INTEGER NOT NULL DEFAULT 0,
    grading_conflict_count  INTEGER NOT NULL DEFAULT 0,
    dual_conflict_count     INTEGER NOT NULL DEFAULT 0,

    -- CV 指标
    total_cells      INTEGER,
    abnormal_cells   INTEGER,
    highest_risk     TEXT,
    max_uncertainty  REAL,

    -- 正文与结构化证据（JSONB 便于按字段查询）
    report_md        TEXT,
    recommendations  JSONB,
    references_list  JSONB,
    cells_json       JSONB,
    knowledge_json   JSONB,
    gradcam_path     TEXT,

    -- DiagnosisState 结构仍在演化：旧记录必须能识别版本
    schema_version   INTEGER     NOT NULL DEFAULT 1
);

COMMENT ON COLUMN diagnosis_reports.operator_id IS 'JWT 中的 uid，不设外键（账号停用后报告仍须保留归属）';
COMMENT ON COLUMN diagnosis_reports.cells_json  IS '精简 cells：不含 probabilities，控制库体积';
COMMENT ON COLUMN diagnosis_reports.schema_version IS 'state 结构版本，解析前先校验';

CREATE INDEX IF NOT EXISTS idx_reports_created
    ON diagnosis_reports (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_operator
    ON diagnosis_reports (operator_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_risk
    ON diagnosis_reports (risk_level);

-- 部分索引：只索引 review_required=true，体积远小于全表索引
CREATE INDEX IF NOT EXISTS idx_reports_review
    ON diagnosis_reports (created_at DESC) WHERE review_required;

-- GIN：可直接查 cells_json，无需全表扫描
CREATE INDEX IF NOT EXISTS idx_reports_cells_gin
    ON diagnosis_reports USING GIN (cells_json jsonb_path_ops);

-- ------------------------------------------------------------
-- 迁移记录
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT        PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO schema_migrations (version) VALUES ('001_init')
    ON CONFLICT (version) DO NOTHING;