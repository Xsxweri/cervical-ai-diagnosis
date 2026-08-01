import type { LucideIcon } from 'lucide-react';

type Tone = 'primary' | 'success' | 'warning' | 'danger' | 'ai' | 'gray';

const TONE_META: Record<Tone, { icon: string; value: string }> = {
  primary: { icon: 'bg-primary-50 text-primary-600', value: 'text-primary-700' },
  success: { icon: 'bg-success-50 text-success-600', value: 'text-success-700' },
  warning: { icon: 'bg-warning-50 text-warning-600', value: 'text-warning-700' },
  danger: { icon: 'bg-danger-50 text-danger-600', value: 'text-danger-700' },
  ai: { icon: 'bg-ai-50 text-ai-600', value: 'text-ai-700' },
  gray: { icon: 'bg-gray-100 text-gray-500', value: 'text-gray-800' },
};

interface Props {
  icon: LucideIcon;
  label: string;
  value: string | number;
  sub?: string;
  tone?: Tone;
}

export default function StatCard({ icon: Icon, label, value, sub, tone = 'primary' }: Props) {
  const meta = TONE_META[tone];
  return (
    <div className="card card-hover p-5 flex items-start gap-4 animate-fade-in">
      <div className={`w-11 h-11 rounded-lg flex items-center justify-center shrink-0 ${meta.icon}`}>
        <Icon size={22} />
      </div>
      <div className="min-w-0">
        <p className="text-xs text-gray-500 font-medium">{label}</p>
        <p className={`text-2xl font-bold leading-tight mt-0.5 truncate ${meta.value}`}>{value}</p>
        {sub && <p className="text-xs text-gray-400 mt-1 truncate">{sub}</p>}
      </div>
    </div>
  );
}