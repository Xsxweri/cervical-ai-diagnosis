import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import DashboardPage from './pages/DashboardPage';
import DiagnosePage from './pages/DiagnosePage';
import AnalyzePage from './pages/AnalyzePage';
import HistoryPage from './pages/HistoryPage';
import LoginPage from './pages/LoginPage';
import NotFoundPage from './pages/NotFoundPage';
import { ToastProvider } from './components/ui/Toast';
import {getToken} from './libts/storage';

function RequireAuth() {
    return getToken() ? <Outlet /> : <Navigate to="/login" replace />;
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
            {/* 未知路径显式提示，不再静默落到仪表盘 */}
            <Route path="*" element={<NotFoundPage />} />
          </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  );
}

export default App;