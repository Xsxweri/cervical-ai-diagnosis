import { Menu, LogOut } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { clearToken } from '../../libts/storage';
import { getDoctor } from '../../libts/storage';

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

export default function Header({ title, onMenuClick }: Props) {
  const navigate = useNavigate();
  const doctor = getDoctor();

  const who = doctor?.name ?? '未识别医生';
  const dept = doctor?.department ?? '';

  const handleLogout = () => {
    if (!window.confirm('确定要退出登录吗？')) return;
    clearToken();
    navigate('/login', { replace: true });
  };

  return (
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
            onClick={handleLogout}
            aria-label="退出登录"
          >
            <LogOut size={16} />
            退出
          </button>
        </div>
      </div>
    </header>
  );
}