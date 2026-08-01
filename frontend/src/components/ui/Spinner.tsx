interface Props {
  size?: number;
  label?: string;
  full?: boolean;
}

export default function Spinner({ size = 36, label, full }: Props) {
  const spinner = (
    <div className="flex flex-col items-center gap-3">
      <div
        className="animate-spin rounded-full border-[3px] border-primary-200 border-t-primary-600"
        style={{ width: size, height: size }}
      />
      {label && <p className="text-sm text-gray-500">{label}</p>}
    </div>
  );

  if (full) {
    return <div className="flex items-center justify-center py-20">{spinner}</div>;
  }
  return spinner;
}