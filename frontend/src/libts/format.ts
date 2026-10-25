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
  'Superficial-Intermediate': '浅层中间层细胞',
  normal_superficial: '表层鳞状细胞',
  normal_intermediate: '中层鳞状细胞',
  normal_columnar: '颈管柱状细胞',
  light_dysplastic: '轻度异型增生 (CIN1)',
  moderate_dysplastic: '中度异型增生 (CIN2)',
  severe_dysplastic: '重度异型增生 (CIN3)',
  carcinoma_in_situ: '原位癌 (CIS)',
  none: '未检出细胞',
};

export function classLabel(cls: string): string {
  return CLASS_LABELS[cls] ?? cls;
}

export type UncertaintyTone = 'low' | 'mid' | 'high';
export interface UncertaintyThresholds {
  high: number;
  medium: number;
}

/**
 * 后端不可用时的最后兜底。
 * 正常情况下应始终使用后端随每个结果返回的 `uncertainty_thresholds`，
 * 避免环境变量调整后前后端判定不一致。
 */
export const UNCERTAINTY_DEFAULT: UncertaintyThresholds = { high: 0.6, medium: 0.3 };

/** 不确定性（归一化熵）→ 三档色调 */
export function uncertaintyTone(
  u: number,
  thresholds: UncertaintyThresholds = UNCERTAINTY_DEFAULT,
): UncertaintyTone {
  if (u > thresholds.high) return 'high';
  if (u > thresholds.medium) return 'mid';
  return 'low';
}

export const UNCERTAINTY_META: Record<UncertaintyTone, { label: string; badge: string; bar: string }> = {
  low: { label: '低', badge: 'bg-success-100 text-success-700', bar: 'bg-success-500' },
  mid: { label: '中', badge: 'bg-warning-100 text-warning-700', bar: 'bg-warning-500' },
  high: { label: '高', badge: 'bg-danger-100 text-danger-700', bar: 'bg-danger-500' },
};

/** 风险等级（基于最大不确定性 + 异常占比） */
export function calcRisk(_maxUncertainty: number, abnormalRatio: number): RiskLevel {
  if (abnormalRatio <= 0) return 'low';
  if (abnormalRatio < 0.3) return 'medium';
  if (abnormalRatio < 0.6) return 'high';
  return 'emergency';
}

export const RISK_META: Record<
  RiskLevel,
  { label: string; badge: string; text: string; dot: string }
> = {
  low: { label: '低风险', badge: 'bg-success-100 text-success-700', text: 'text-success-600', dot: 'bg-success-500' },
  medium: { label: '中风险', badge: 'bg-warning-100 text-warning-700', text: 'text-warning-600', dot: 'bg-warning-500' },
  high: { label: '高风险', badge: 'bg-danger-100 text-danger-700', text: 'text-danger-600', dot: 'bg-danger-500' },
  emergency: { label: '紧急', badge: 'bg-danger-600 text-white', text: 'text-danger-600', dot: 'bg-danger-600' },
  unreadable: { label: '无法判读', badge: 'bg-gray-200 text-gray-700', text: 'text-gray-500', dot: 'bg-gray-400' },
};

export const RISK_ORDER: RiskLevel[] = ['low', 'medium', 'high', 'emergency'];

/** 取多个结果中的最高风险（忽略 unreadable，避免把"未检出"抬高） */
export function maxRisk(levels: RiskLevel[]): RiskLevel {
  return levels
    .filter((l) => l !== 'unreadable')
    .reduce<RiskLevel>(
      (acc, cur) => (RISK_ORDER.indexOf(cur) > RISK_ORDER.indexOf(acc) ? cur : acc),
      'low',
    );
}

/** 时间戳 → 本地时间字符串 */
export function formatTime(ts: number): string {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function riskFromText(text: string): RiskLevel {
  if (/紧急|立即就诊|急诊|高危/.test(text)) return 'emergency';
  if (/高风|高度怀疑|HSIL|CIN ?2|CIN ?3|癌/.test(text)) return 'high';
  if (/中风险|ASC-US|LSIL|低度|CIN ?1|可疑/.test(text)) return 'medium';
  return 'low';
}