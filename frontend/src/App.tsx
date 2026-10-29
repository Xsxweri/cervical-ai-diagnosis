import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import DashboardPage from './pages/DashboardPage';
import DiagnosePage from './pages/DiagnosePage';
import AnalyzePage from './pages/AnalyzePage';
import HistoryPage from './pages/HistoryPage';
import LoginPage from './pages/LoginPage';
import NotFoundPage from './pages/NotFoundPage';
import AccountsPage from './pages/AccountsPage';
import Spinner from './components/ui/Spinner';
import { ToastProvider } from './components/ui/Toast';
import {getToken} from './libts/storage';
import { useMe } from './libts/session';

function RequireAuth() {
    return getToken() ? <Outlet /> : <Navigate to="/login" replace />;
}

function RequireSysAdmin() {
  const { me, loading, error } = useMe();
  if (loading && !me) return <Spinner full label="正在校验权限…" />;
  if (!me) {
    return error ? <Navigate to="/" replace /> : null;
  }
  return me.role === 'sysadmin' ? <Outlet /> : <Navigate to="/" replace />;
}

function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route element={<RequireAuth />}>
           <Route element={<AppLayout />}>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/diagnose" element={<DiagnosePage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/history" element={<HistoryPage />} />
            {/* 账号管理仅管理员可见；后端 require_admin 才是真正的边界 */}
            <Route element={<RequireSysAdmin />}>
              <Route path="/accounts" element={<AccountsPage />} />
            </Route>
            <Route path="*" element={<NotFoundPage />} />
          </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  );
}

export default App;