import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { CheckCircle2, AlertTriangle, XCircle, X } from 'lucide-react';
import { ToastContext } from '../../hooks/useToast';
import type {
  ToastAction,
  ToastContextValue,
  ToastOptions,
  ToastType,
} from '../../hooks/useToast';

interface ToastItem {
  id: number;
  type: ToastType;
  message: string;
  action?: ToastAction;
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

const DEFAULT_DURATION = 4000;
const ACTION_DURATION = 6000;

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
    (type: ToastType, message: string, opts?: ToastOptions) => {
      const id = ++idRef.current;
      setToasts((prev) => [
        ...prev,
        { id, type, message, action: opts?.action },
      ]);
      const ms =
        opts?.duration ?? (opts?.action ? ACTION_DURATION : DEFAULT_DURATION);
      timersRef.current.push(window.setTimeout(() => remove(id), ms));
    },
    [remove]
  );

  const value: ToastContextValue = {
    success: (m, o) => push('success', m, o),
    error: (m, o) => push('error', m, o),
    info: (m, o) => push('info', m, o),
  };

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        className="fixed top-4 right-4 z-[100] space-y-2 w-80 max-w-[calc(100vw-2rem)]"
        role="status"
        aria-live="polite"
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            className="relative card p-3 flex items-start gap-2.5 animate-fade-in shadow-panel"
          >
            {ICONS[t.type]}
            <p className="flex-1 text-[13px] text-ink leading-snug break-words">
              {t.message}
            </p>
            {t.action && (
              <button
                className="shrink-0 text-[12.5px] font-medium text-hematoxylin hover:underline"
                onClick={() => {
                  t.action?.onClick();
                  remove(t.id);
                }}
              >
                {t.action.label}
              </button>
            )}
            <button
              className="text-muted hover:text-ink shrink-0"
              onClick={() => remove(t.id)}
              aria-label="关闭提示"
            >
              <X size={15} />
            </button>
            <span
              className={`absolute bottom-0 left-0 h-0.5 ${BAR[t.type]}`}
              style={{ width: '100%' }}
            />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}