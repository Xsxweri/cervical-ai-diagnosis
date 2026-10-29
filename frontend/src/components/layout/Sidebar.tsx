import { NavLink } from 'react-router-dom';
import { History, LayoutDashboard, ScanSearch, ShieldCheck, Sparkles, X } from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', label: '判读台', icon: LayoutDashboard, end: true },
  { to: '/diagnose', label: '智能诊断', icon: Sparkles, end: false },
  { to: '/analyze', label: '快速分析', icon: ScanSearch, end: false },
  { to: '/history', label: '本机记录', icon: History, end: false },
];

/** 载玻片抽象标记：圆角矩形 + 一角标签块 */
function BrandMark({ className = '' }: { className?: string }) {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true" className={className}>
      <rect x="2.5" y="5.5" width="19" height="13" rx="2.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M14.5 5.5H19a2.5 2.5 0 0 1 2.5 2.5v2H14.5V5.5Z" fill="currentColor" opacity="0.85" />
    </svg>
  );
}

interface Props {
  open: boolean;
  onClose: () => void;
  isSysAdmin?: boolean;
}

export default function Sidebar({ open, onClose, isSysAdmin=false }: Props) {
  const items = isSysAdmin
    ? [...NAV_ITEMS, { to: '/accounts', label: '账号管理', icon: ShieldCheck, end: false }]
    : NAV_ITEMS;
   return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-40 bg-ink/40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 w-60 bg-white border-r border-hairline flex flex-col transform transition-transform duration-200 lg:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* 品牌 */}
        <div className="flex items-center gap-3 px-5 h-14 border-b border-hairline">
          <BrandMark className="shrink-0 text-hematoxylin" />
          <div className="min-w-0">
            <p className="font-serif text-[14px] font-semibold leading-tight text-ink truncate">
              宫颈细胞学
            </p>
            <p className="text-[11px] tracking-[0.12em] text-muted leading-tight">辅助判读系统</p>
          </div>
          <button
            className="ml-auto lg:hidden text-muted hover:text-ink"
            onClick={onClose}
            aria-label="关闭菜单"
          >
            <X size={20} />
          </button>
        </div>

        {/* 导航 */}
        <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto" aria-label="主导航">
          {items.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors ${
                  isActive
                    ? 'bg-primary-50 text-hematoxylin font-medium border-l-2 border-hematoxylin'
                    : 'text-muted hover:bg-glass hover:text-ink border-l-2 border-transparent'
                }`
              }
            >
              <Icon size={17} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* 免责声明 */}
        <div className="px-5 py-4 border-t border-hairline">
          <p className="text-[11.5px] leading-relaxed text-muted">
            本系统输出仅供辅助诊断参考，
            <br />
            最终诊断以临床医生判断为准。
          </p>
        </div>
      </aside>
    </>
  );
}