import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { CheckCircle2, AlertTriangle, XCircle, X } from 'lucide-react';
import { ToastContext } from '../../hooks/useToast';
import type { ToastContextValue, ToastType } from '../../hooks/useToast';

interface ToastItem {
  id: number;
  type: ToastType;
  message: string;
}

const ICONS: Record<ToastType, ReactNode> = {
  success: <CheckCircle2 size={18} className="text-success-500 shrink-0" />,
  error: <XCircle size={18} className="text-danger-500 shrink-0" />,
  info: <AlertTriangle size={18} className="text-warning-500 shrink-0" />,
};

const BAR: Record<ToastType, string> = {
  success: 'bg-success-500',
  error: 'bg-danger-500',
  info: 'bg-warning-500',
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const idRef = useRef(0);
  const timersRef = useRef<number[]>([]);

  // 卸载时清理所有自动关闭定时器，避免回调在组件销毁后触发
  useEffect(() => {
    return () => {
      timersRef.current.forEach((t) => window.clearTimeout(t));
      timersRef.current = [];
    };
  }, []);

  const remove = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (type: ToastType, message: string) => {
      const id = ++idRef.current;
      setToasts((prev) => [...prev, { id, type, message }]);
      timersRef.current.push(window.setTimeout(() => remove(id), 4000));
    },
    [remove]
  );

  const value: ToastContextValue = {
    success: (m) => push('success', m),
    error: (m) => push('error', m),
    info: (m) => push('info', m),
  };

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="fixed top-4 right-4 z-[100] space-y-2 w-80 max-w-[calc(100vw-2rem)]">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="relative card p-3 flex items-start gap-2.5 animate-fade-in shadow-lg border-gray-200"
          >
            {ICONS[t.type]}
            <p className="flex-1 text-sm text-gray-700 leading-snug break-words">{t.message}</p>
            <button
              className="text-gray-400 hover:text-gray-600 shrink-0"
              onClick={() => remove(t.id)}
              aria-label="关闭"
            >
              <X size={15} />
            </button>
            <span className={`absolute bottom-0 left-0 h-0.5 ${BAR[t.type]}`} style={{ width: '100%' }} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
