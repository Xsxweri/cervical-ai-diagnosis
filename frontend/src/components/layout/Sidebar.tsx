import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Sparkles,
  ScanSearch,
  History,
  X,
  Microscope,
} from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', label: '仪表盘', icon: LayoutDashboard, end: true },
  { to: '/diagnose', label: '智能诊断', icon: Sparkles, end: false },
  { to: '/analyze', label: '快速分析', icon: ScanSearch, end: false },
  { to: '/history', label: '诊断历史', icon: History, end: false },
];

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function Sidebar({ open, onClose }: Props) {
  return (
    <>
      {/* 移动端遮罩 */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-gray-900/40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-gray-100 flex flex-col transform transition-transform duration-200 lg:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Logo 区 */}
        <div className="flex items-center gap-3 px-5 h-16 border-b border-gray-100">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-primary-600 to-primary-400 flex items-center justify-center text-white shrink-0">
            <Microscope size={20} />
          </div>
          <div className="min-w-0">
            <p className="font-semibold text-gray-900 leading-tight truncate">
              AI-Agent 宫颈病理诊断
            </p>
            <p className="text-[11px] text-gray-400 leading-tight">v2.1 · YOLOv8 + ResNet50 + LangGraph</p>
          </div>
          <button
            className="ml-auto lg:hidden text-gray-400 hover:text-gray-600"
            onClick={onClose}
            aria-label="关闭菜单"
          >
            <X size={20} />
          </button>
        </div>

        {/* 导航 */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
            >
              <Icon size={18} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* 底部信息 */}
        <div className="px-5 py-4 border-t border-gray-100">
          <p className="text-xs text-gray-400 leading-relaxed">
            本系统输出仅供辅助诊断参考，
            <br />
            最终诊断以临床医生判断为准。
          </p>
        </div>
      </aside>
    </>
  );
}