import { createContext, useContext } from 'react';

/** Toast 类型 */
export type ToastType = 'success' | 'error' | 'info';

/** Toast 上的可选动作，例如「撤销」 */
export interface ToastAction {
  label: string;
  onClick: () => void;
}

export interface ToastOptions {
  action?: ToastAction;
  /** 毫秒；带 action 时默认 6000，否则 4000 */
  duration?: number;
}

export interface ToastContextValue {
  success: (msg: string, opts?: ToastOptions) => void;
  error: (msg: string, opts?: ToastOptions) => void;
  info: (msg: string, opts?: ToastOptions) => void;
}

export const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast 必须在 ToastProvider 内使用');
  return ctx;
}