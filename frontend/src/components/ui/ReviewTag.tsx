/**
 * 待复核标记。
 *
 * 触发条件：存在「二分类判异常、分级模型归为非异常」的冲突细胞。
 * 这类病例不得给出确定的病例级分级，必须显式提示人工复核。
 */
export default function ReviewTag({
  required,
  count,
}: {
  required?: boolean;
  count?: number;
}) {
  if (!required) return null;
  const n = count ?? 0;
  return (
    <span
      className="badge bg-danger-100 text-danger-800"
      title={
        n > 0
          ? `存在 ${n} 个结论冲突细胞（二分类与分级模型判定不一致），需人工镜检复核`
          : '存在结论冲突，需人工镜检复核'
      }
    >
      待复核{n > 0 ? ` · ${n}` : ''}
    </span>
  );
}