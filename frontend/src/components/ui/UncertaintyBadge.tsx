import { UNCERTAINTY_META, pct, uncertaintyTone } from '../../libts/format';
import type { UncertaintyThresholds } from '../../libts/format';

interface Props {
  value: number;
  showLabel?: boolean;
  /** 优先使用后端结果里的 uncertainty_thresholds，避免阈值调整后前后端不一致 */
  thresholds?: UncertaintyThresholds;
}

export default function UncertaintyBadge({ value, showLabel = true, thresholds }: Props) {
  const tone = uncertaintyTone(value, thresholds);
  const meta = UNCERTAINTY_META[tone];
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${meta.bar} transition-all duration-500`}
          style={{ width: `${Math.min(value * 100, 100)}%` }}
        />
      </div>
      <span className={`badge ${meta.badge}`}>
        {pct(value)}
        {showLabel && <span className="opacity-70">· {meta.label}</span>}
      </span>
    </div>
  );
}