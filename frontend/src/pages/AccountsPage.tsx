import { useCallback, useEffect, useMemo, useState } from 'react';
import { KeyRound, Search, ShieldCheck, Unlock, UserPlus, Users } from 'lucide-react';
import EmptyState from '../components/ui/EmptyState';
import Modal from '../components/ui/Modal';
import Spinner from '../components/ui/Spinner';
import { useToast } from '../hooks/useToast';
import { errText, useMe } from '../libts/session';
import {
  createAccount,
  fetchAccounts,
  resetAccountPassword,
  unlockAccount,
  updateAccount,
} from '../libts/api';
import type { AssignableRole, UserPublic, UserRole } from '../types';

const MIN_PWD = 8;                 // 与后端 api/schemas.py MIN_PASSWORD_LEN 对齐
const USERNAME_RE = /^[A-Za-z0-9_.-]{3,64}$/;

const ROLE_LABEL: Record<UserRole, string> = {
  admin: '管理员',
  doctor: '医生',
  sysadmin: '系统管理员',
};

const toAssignable = (r: UserRole): AssignableRole => (r === 'sysadmin' ? 'admin' : r);

interface FormState {
  username: string;
  password: string;
  name: string;
  department: string;
  hospital: string;
  role: AssignableRole;
}

const EMPTY_FORM: FormState = {
  username: '', password: '', name: '', department: '', hospital: '', role: 'doctor',
};

const fmt = (iso?: string | null) => (iso ? new Date(iso).toLocaleString('zh-CN') : '—');

const isLocked = (u: UserPublic) =>
  !!u.locked_until && new Date(u.locked_until).getTime() > Date.now();

function statusOf(u: UserPublic): { text: string; cls: string } {
  if (!u.is_active) return { text: '已停用', cls: 'bg-gray-100 text-gray-500' };
  if (isLocked(u)) return { text: '锁定中', cls: 'bg-amber-50 text-amber-700' };
  return { text: '正常', cls: 'bg-green-50 text-green-700' };
}

export default function AccountsPage() {
  const { me } = useMe();
  const toast = useToast();

  const [users, setUsers] = useState<UserPublic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');

  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<UserPublic | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [busy, setBusy] = useState(false);

  const [pwdTarget, setPwdTarget] = useState<UserPublic | null>(null);
  const [pwd, setPwd] = useState('');

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setUsers(await fetchAccounts());
      setError(null);
    } catch (e: unknown) {
      setError(errText(e, '账号列表读取失败'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return users;
    return users.filter(
      (u) =>
        u.username.toLowerCase().includes(q) ||
        (u.name ?? '').toLowerCase().includes(q) ||
        (u.department ?? '').toLowerCase().includes(q),
    );
  }, [users, query]);

  const openCreate = () => {
    setEditing(null);
    setForm(EMPTY_FORM);
    setFormOpen(true);
  };

  const openEdit = (u: UserPublic) => {
    setEditing(u);
    setForm({
      username: u.username,
      password: '',
      name: u.name ?? '',
      department: u.department ?? '',
      hospital: u.hospital ?? '',
      role: toAssignable(u.role),
    });
    setFormOpen(true);
  };

  const submit = async () => {
    if (!editing) {
      if (!USERNAME_RE.test(form.username)) {
        toast.error('用户名需 3–64 位，仅字母、数字、下划线、点、短横线');
        return;
      }
      if (form.password.length < MIN_PWD) {
        toast.error(`密码至少 ${MIN_PWD} 位`);
        return;
      }
    }
    setBusy(true);
    try {
      if (editing) {
        await updateAccount(editing.id, {
          name: form.name || undefined,
          department: form.department || undefined,
          hospital: form.hospital || undefined,
          ...(editing.role === 'sysadmin' ? {} : { role: form.role }),
        });
        toast.success('已保存');
      } else {
        await createAccount({
          username: form.username,
          password: form.password,
          name: form.name || undefined,
          department: form.department || undefined,
          hospital: form.hospital || undefined,
          role: form.role,
        });
        toast.success('账号已创建');
      }
      setFormOpen(false);
      await reload();
    } catch (e: unknown) {
      toast.error(errText(e, '保存失败'));
    } finally {
      setBusy(false);
    }
  };

  const toggleActive = async (u: UserPublic) => {
    if (u.id === me?.id) return;
    if (u.is_active && !window.confirm(`停用 ${u.username}？该账号将无法登录。`)) return;
    try {
      await updateAccount(u.id, { is_active: !u.is_active });
      toast.success(u.is_active ? '已停用' : '已启用');
      await reload();
    } catch (e: unknown) {
      toast.error(errText(e, '操作失败'));
    }
  };

  const doUnlock = async (u: UserPublic) => {
    try {
      await unlockAccount(u.id);
      toast.success(`已解锁 ${u.username}`);
      await reload();
    } catch (e: unknown) {
      toast.error(errText(e, '解锁失败'));
    }
  };

  const doResetPwd = async () => {
    if (!pwdTarget) return;
    if (pwd.length < MIN_PWD) {
      toast.error(`密码至少 ${MIN_PWD} 位`);
      return;
    }
    setBusy(true);
    try {
      await resetAccountPassword(pwdTarget.id, pwd);
      toast.success(`已重置 ${pwdTarget.username} 的密码`);
      setPwdTarget(null);
      setPwd('');
    } catch (e: unknown) {
      toast.error(errText(e, '重置失败'));
    } finally {
      setBusy(false);
    }
  };

  if (loading && users.length === 0) return <Spinner full label="正在读取账号…" />;

  if (error) {
    return (
      <EmptyState
        icon={ShieldCheck}
        title="无法读取账号列表"
        description={error}
        action={
          <button className="btn btn-secondary" onClick={() => void reload()}>
            重试
          </button>
        }
      />
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
          <input
            className="input pl-9 w-64"
            placeholder="搜索用户名 / 姓名 / 科室"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <span className="text-[12.5px] text-muted tabular">
          共 {users.length} 个账号，启用 {users.filter((u) => u.is_active).length} 个
        </span>
        <button className="btn btn-primary ml-auto" onClick={openCreate}>
          <UserPlus size={16} />
          新建账号
        </button>
      </div>

      {filtered.length === 0 ? (
        <EmptyState icon={Users} title="没有匹配的账号" description="换个关键词，或新建一个账号。" />
      ) : (
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-glass text-muted">
              <tr className="text-left">
                <th className="px-4 py-3 font-medium">账号</th>
                <th className="px-4 py-3 font-medium">姓名 / 科室</th>
                <th className="px-4 py-3 font-medium">角色</th>
                <th className="px-4 py-3 font-medium">状态</th>
                <th className="px-4 py-3 font-medium">最近登录</th>
                <th className="px-4 py-3 font-medium text-right">操作</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((u) => {
                const st = statusOf(u);
                const self = u.id === me?.id;
                return (
                  <tr key={u.id} className="border-t border-hairline">
                    <td className="px-4 py-3">
                      <span className="font-medium text-ink">{u.username}</span>
                      {self && <span className="ml-2 text-[11px] text-hematoxylin">（本人）</span>}
                    </td>
                    <td className="px-4 py-3 text-muted">
                      {u.name || '—'}
                      {u.department && <span className="text-muted/70"> · {u.department}</span>}
                    </td>
                    <td className="px-4 py-3">{ROLE_LABEL[u.role] ?? u.role}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-[12px] ${st.cls}`}>{st.text}</span>
                      {u.failed_attempts > 0 && (
                        <span className="ml-2 text-[11px] text-muted">失败 {u.failed_attempts} 次</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted tabular">{fmt(u.last_login_at)}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end gap-1">
                        <button className="btn btn-ghost" onClick={() => openEdit(u)}>
                          编辑
                        </button>
                        <button
                          className="btn btn-ghost"
                          onClick={() => {
                            setPwdTarget(u);
                            setPwd('');
                          }}
                        >
                          <KeyRound size={14} />
                          重置密码
                        </button>
                        {isLocked(u) && (
                          <button className="btn btn-ghost" onClick={() => void doUnlock(u)}>
                            <Unlock size={14} />
                            解锁
                          </button>
                        )}
                        <button
                          className="btn btn-ghost disabled:opacity-40"
                          disabled={self || u.role === 'sysadmin'}
                          title={self ? '不能停用自己的账号' : u.role === 'sysadmin' ? '系统管理员需由运维停用' : undefined}
                          onClick={() => void toggleActive(u)}
                        >
                          {u.is_active ? '停用' : '启用'}
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* 新建 / 编辑 */}
      <Modal
        open={formOpen}
        onClose={() => setFormOpen(false)}
        title={editing ? `编辑账号 · ${editing.username}` : '新建账号'}
        maxWidth="max-w-lg"
      >
        <div className="space-y-4">
          <div>
            <label className="label">用户名</label>
            <input
              className="input"
              value={form.username}
              disabled={!!editing}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
            />
            {editing && <p className="hint">用户名是登录键且已写入审计日志，不可修改。</p>}
          </div>

          {!editing && (
            <div>
              <label className="label">初始密码</label>
              <input
                className="input"
                type="password"
                autoComplete="new-password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
              />
              <p className="hint">至少 {MIN_PWD} 位。</p>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label">姓名</label>
              <input
                className="input"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </div>
            <div>
              <label className="label">科室</label>
              <input
                className="input"
                value={form.department}
                onChange={(e) => setForm({ ...form, department: e.target.value })}
              />
            </div>
          </div>

          <div>
            <label className="label">医院</label>
            <input
              className="input"
              value={form.hospital}
              onChange={(e) => setForm({ ...form, hospital: e.target.value })}
            />
          </div>

          <div>
            <label className="label">角色</label>
            <select
              className="input"
              value={form.role}
              disabled={editing?.role === 'sysadmin'}
              onChange={(e) => setForm({ ...form, role: e.target.value as AssignableRole })}
            >
              <option value="doctor">医生</option>
              <option value="admin">管理员</option>
            </select>
            {editing?.role === 'sysadmin' && (
              <p className="hint">系统管理员身份只能由运维在数据库中调整，界面不予变更。</p>
            )}
            {editing?.id === me?.id && (
              <p className="hint">不能撤销自己的系统管理员权限（后端会拒绝）。</p>
            )}
          </div>

          <div className="flex justify-end gap-2 pt-2">
            <button className="btn btn-secondary" onClick={() => setFormOpen(false)}>
              取消
            </button>
            <button className="btn btn-primary" disabled={busy} onClick={() => void submit()}>
              {busy ? '保存中…' : '保存'}
            </button>
          </div>
        </div>
      </Modal>

      {/* 管理员重置密码 */}
      <Modal
        open={!!pwdTarget}
        onClose={() => setPwdTarget(null)}
        title={`重置密码 · ${pwdTarget?.username ?? ''}`}
        maxWidth="max-w-md"
      >
        <div className="space-y-4">
          <div>
            <label className="label">新密码</label>
            <input
              className="input"
              type="password"
              autoComplete="new-password"
              value={pwd}
              onChange={(e) => setPwd(e.target.value)}
            />
            <p className="hint">至少 {MIN_PWD} 位。告知本人后请提醒其自行修改。</p>
          </div>
          <div className="flex justify-end gap-2">
            <button className="btn btn-secondary" onClick={() => setPwdTarget(null)}>
              取消
            </button>
            <button className="btn btn-primary" disabled={busy} onClick={() => void doResetPwd()}>
              {busy ? '提交中…' : '确认重置'}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}