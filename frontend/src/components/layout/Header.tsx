import { useState } from 'react';
import { KeyRound, LogOut, Menu } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Modal from '../ui/Modal';
import { useToast } from '../../hooks/useToast';
import { changeOwnPassword } from '../../libts/api';
import { useMe,errText } from '../../libts/session';
import { clearToken, getDoctor } from '../../libts/storage';

interface Props {
  title: string;
  onMenuClick: () => void;
}

const dateFormatter = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  weekday: 'short',
});

const MIN_PWD = 8;

export default function Header({ title, onMenuClick }: Props) {
  const navigate = useNavigate();
  const doctor = getDoctor();
  const toast = useToast();

  const [pwdOpen, setPwdOpen] = useState(false);
  const [oldPwd, setOldPwd] = useState('');
  const [newPwd, setNewPwd] = useState('');
  const [confirmPwd, setConfirmPwd] = useState('');
  const [busy, setBusy] = useState(false);

  const { me } = useMe();
  const who = me?.name ?? doctor?.name ?? '未识别医生';
  const dept = me?.department ?? doctor?.department ?? '';

  const handleLogout = () => {
    if (!window.confirm('确定要退出登录吗？')) return;
    clearToken();
    navigate('/login', { replace: true });
  };

  const openPwd = () => {
    setOldPwd('');
    setNewPwd('');
    setConfirmPwd('');
    setPwdOpen(true);
  };

  const submitPwd = async () => {
    if (!oldPwd || !newPwd) {
      toast.error('请填写原密码与新密码');
      return;
    }
    if (newPwd.length < MIN_PWD) {
      toast.error(`新密码至少 ${MIN_PWD} 位`);
      return;
    }
    if (newPwd !== confirmPwd) {
      toast.error('两次输入的新密码不一致');
      return;
    }
    if (newPwd === oldPwd) {
      toast.error('新密码不能与原密码相同');
      return;
    }
  
  setBusy(true);
    try {
      await changeOwnPassword(oldPwd, newPwd);
      setPwdOpen(false);
      /**
       * JWT 无状态，改密不会让旧 token 失效（core/auth.py
       * ACCESS_TOKEN_EXPIRE_HOURS=1）。必须主动清 token 重新登录，
       * 否则用户会以为"改完还是老密码能进"。
       */
      clearToken();
      toast.success('密码已修改，请用新密码重新登录');
      navigate('/login', { replace: true });
    } catch (e: unknown) {
      toast.error(errText(e, '修改失败'));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <header className="sticky top-0 z-30 bg-white/90 backdrop-blur border-b border-hairline">
        <div className="flex items-center gap-3 h-14 px-4 sm:px-6 lg:px-8 w-full">
          <button
            className="lg:hidden text-muted hover:text-ink"
            onClick={onMenuClick}
            aria-label="打开菜单"
          >
            <Menu size={22} />
          </button>

          <h1 className="text-[15px] font-semibold text-ink truncate">{title}</h1>

          <div className="ml-auto flex items-center gap-4 shrink-0">
            <span className="hidden sm:inline text-[12.5px] text-muted tabular">
              {dateFormatter.format(new Date())}
            </span>
            <span className="text-[12.5px] text-ink">
              {who}
              {dept && <span className="text-muted"> · {dept}</span>}
            </span>
            <button
              className="flex items-center gap-1 text-[12.5px] text-muted hover:text-ink"
              onClick={openPwd}
              aria-label="修改密码"
            >
              <KeyRound size={15} />
              改密
            </button>
            <button
              className="flex items-center gap-1 text-[12.5px] text-muted hover:text-ink"
              onClick={handleLogout}
              aria-label="退出登录"
            >
              <LogOut size={16} />
              退出
            </button>
          </div>
        </div>
      </header>

      <Modal
        open={pwdOpen}
        onClose={() => setPwdOpen(false)}
        title="修改密码"
        maxWidth="max-w-md"
      >
        <div className="space-y-4">
          <div>
            <label className="label">原密码</label>
            <input
              className="input"
              type="password"
              autoComplete="current-password"
              value={oldPwd}
              onChange={(e) => setOldPwd(e.target.value)}
            />
          </div>
          <div>
            <label className="label">新密码</label>
            <input
              className="input"
              type="password"
              autoComplete="new-password"
              value={newPwd}
              onChange={(e) => setNewPwd(e.target.value)}
            />
            <p className="hint">至少 {MIN_PWD} 位。</p>
          </div>
          <div>
            <label className="label">确认新密码</label>
            <input
              className="input"
              type="password"
              autoComplete="new-password"
              value={confirmPwd}
              onChange={(e) => setConfirmPwd(e.target.value)}
            />
          </div>
          <p className="hint">修改成功后需要重新登录。</p>
          <div className="flex justify-end gap-2 pt-2">
            <button className="btn btn-secondary" onClick={() => setPwdOpen(false)}>
              取消
            </button>
            <button className="btn btn-primary" disabled={busy} onClick={() => void submitPwd()}>
              {busy ? '提交中…' : '确认修改'}
            </button>
          </div>
        </div>
      </Modal>
    </>
  );
}