import { UNCERTAINTY_META, pct, uncertaintyTone } from '../../libts/format';

interface Props {
  value: number;
  showLabel?: boolean;
}

export default function UncertaintyBadge({ value, showLabel = true }: Props) {
  const tone = uncertaintyTone(value);
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