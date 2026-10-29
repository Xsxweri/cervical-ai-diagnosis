\set ON_ERROR_STOP on

-- 内联 CHECK 的自动名称为 <table>_<column>_check
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users ADD CONSTRAINT users_role_check
    CHECK (role IN ('doctor', 'admin', 'sysadmin'));

-- 不自动授予任何人 sysadmin：现有 admin 一律降为 doctor，
-- 由运维显式指定第一个系统管理员（见下方 bootstrap）
UPDATE users SET role = 'doctor' WHERE role = 'admin';

INSERT INTO schema_migrations (version) VALUES ('002_sysadmin')
    ON CONFLICT (version) DO NOTHING;