import type { RiskLevel } from '../../types';

export default function RiskScopeTag({
  scope,
  level,
}: {
  scope?: 'cell' | 'case' | null;
  level?: RiskLevel | null;
}) {
  if (!scope) return null;
  const cell = scope === 'cell';
  const urgent = level === 'high' || level === 'emergency';
  return (
    <span
      className={`badge ${
        cell
          ? urgent
            ? 'bg-warning-100 text-warning-800'
            : 'bg-gray-100 text-gray-600'
          : 'bg-primary-50 text-primary-700'
      }`}
      title={cell ? '仅基于少量细胞的提示，不代表病例整体风险' : '基于异常细胞占比的病例级结论'}
    >
      {cell ? '细胞级提示' : '病例级'}
    </span>
  );
}