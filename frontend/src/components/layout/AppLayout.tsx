import { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';

const TITLES: Record<string, string> = {
  '/': '判读台',
  '/diagnose': '智能诊断',
  '/analyze': '快速分析',
  '/history': '本机最近记录',
};

export default function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { pathname } = useLocation();
  const title = TITLES[pathname] ?? '判读台';

  return (
    <div className="min-h-screen bg-glass">
      {/* 键盘用户跳过导航 */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[200] focus:rounded-md focus:bg-hematoxylin focus:px-4 focus:py-2 focus:text-sm focus:text-white"
      >
        跳到主要内容
      </a>

      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="lg:pl-60 min-h-screen flex flex-col">
        <Header title={title} onMenuClick={() => setSidebarOpen(true)} />
        <main
          id="main"
          tabIndex={-1}
          className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 focus:outline-none"
        >
          <Outlet />
        </main>
      </div>
    </div>
  );
}