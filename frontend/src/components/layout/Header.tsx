import { Menu } from 'lucide-react';

interface Props {
  title: string;
  onMenuClick: () => void;
}

export default function Header({ title, onMenuClick }: Props) {
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
        <h1 className="text-base sm:text-lg font-semibold text-gray-900 truncate">{title}</h1>
      </div>
    </header>
  );
}