import { RISK_META } from '../../libts/format';
import type { RiskLevel } from '../../types';

interface Props {
  level: RiskLevel;
  size?: 'sm' | 'md';
}

export default function RiskBadge({ level, size = 'sm' }: Props) {
  const meta = RISK_META[level];
  return (
    <span
      className={`badge ${meta.badge} ${size === 'md' ? 'px-3 py-1 text-sm' : ''}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${meta.dot}`} />
      {meta.label}
    </span>
  );
}