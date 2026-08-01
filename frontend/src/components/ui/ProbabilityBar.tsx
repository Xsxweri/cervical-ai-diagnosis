import { pct } from '../../libts/format';

interface Props {
  label: string;
  value: number;
  colorClass?: string;
}

export default function ProbabilityBar({ label, value, colorClass = 'bg-primary-500' }: Props) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-32 text-gray-600 truncate shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${colorClass} transition-all duration-500`}
          style={{ width: `${Math.min(value * 100, 100)}%` }}
        />
      </div>
      <span className="w-12 text-right text-gray-500 font-medium shrink-0">{pct(value)}</span>
    </div>
  );
}