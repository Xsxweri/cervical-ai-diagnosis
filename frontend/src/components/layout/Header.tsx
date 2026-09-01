import { Menu, LogOut } from 'lucide-react';
import {useNavigate} from 'react-router-dom';
import {clearToken} from '../../libts/storage';

interface Props {
  title: string;
  onMenuClick: () => void;
}

export default function Header({ title, onMenuClick }: Props) {
  const navigate = useNavigate();

  const handleLogout = () => {
      if (!window.confirm('确定要退出登录吗？')) return;
      clearToken();
      navigate('/login', { replace: true });
  };

  return (
    <header className="sticky top-0 z-30 bg-white/80 backdrop-blur border-b border-gray-100">
      <div className="flex items-center gap-3 h-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto w-full">
        <button
          className="lg:hidden text-gray-500 hover:text-gray-700"
          onClick={onMenuClick}
          aria-label="打开菜单"
        >
          <Menu size={22} />
        </button>
        <h1 className="text-base sm:text-lg font-semibold text-gray-900 truncate flex-1 min-w-0">{title}</h1>
       <button
          className="ml-auto flex items-center gap-1 text-sm text-gray-500 hover:text-gray-800 shrink-0"
          onClick={handleLogout}
          aria-label="退出登录"
        >
          <LogOut size={18} />
          退出
        </button>
      </div>
    </header>
  );
}