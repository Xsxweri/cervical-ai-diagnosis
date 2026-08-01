import type { RiskLevel } from '../types';

/** 百分比格式化：0.974 → "97.4%" */
export function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return '--';
  return `${(v * 100).toFixed(digits)}%`;
}

/** 英文字符串类名 → 中文 */
export const CLASS_LABELS: Record<string, string> = {
  normal: '正常细胞',
  abnormal: '异常细胞',
  Dyskeratotic: '角化不良细胞',
  Koilocytotic: '挖空细胞',
  Metaplastic: '化生细胞',
  Parabasal: '副基底层细胞',
  SuperficialIntermediate: '浅层中间层细胞',
};

export function classLabel(cls: string): string {
  return CLASS_LABELS[cls] ?? cls;
}

/** 不确定性 → 三档色调（阈值 0.15 / 0.3） */
export type UncertaintyTone = 'low' | 'mid' | 'high';
export function uncertaintyTone(u: number): UncertaintyTone {
  if (u > 0.3) return 'high';
  if (u > 0.15) return 'mid';
  return 'low';
}

export const UNCERTAINTY_META: Record<UncertaintyTone, { label: string; badge: string; bar: string }> = {
  low: { label: '低', badge: 'bg-success-100 text-success-700', bar: 'bg-success-500' },
  mid: { label: '中', badge: 'bg-warning-100 text-warning-700', bar: 'bg-warning-500' },
  high: { label: '高', badge: 'bg-danger-100 text-danger-700', bar: 'bg-danger-500' },
};

/** 风险等级（基于最大不确定性 + 异常占比） */
export function calcRisk(maxUncertainty: number, abnormalRatio: number): RiskLevel {
  if (maxUncertainty > 0.3) return 'emergency';
  if (abnormalRatio >= 0.5) return 'high';
  if (abnormalRatio > 0) return 'medium';
  return 'low';
}

export const RISK_META: Record<
  RiskLevel,
  { label: string; badge: string; text: string; dot: string }
> = {
  low: { label: '低风险', badge: 'bg-success-100 text-success-700', text: 'text-success-600', dot: 'bg-success-500' },
  medium: { label: '中风险', badge: 'bg-warning-100 text-warning-700', text: 'text-warning-600', dot: 'bg-warning-500' },
  high: { label: '高风险', badge: 'bg-danger-100 text-danger-700', text: 'text-danger-600', dot: 'bg-danger-500' },
  emergency: { label: '紧急', badge: 'bg-danger-600 text-white', text: 'text-danger-600', dot: 'bg-danger-600' },
};

/** 时间戳 → 本地时间字符串 */
export function formatTime(ts: number): string {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}