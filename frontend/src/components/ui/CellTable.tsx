import { classLabel, pct } from '../../libts/format';
import type { CellResult } from '../../types';
import UncertaintyBadge from './UncertaintyBadge';

interface Props {
  cells: CellResult[];
  limit?: number;
}

export default function CellTable({ cells, limit }: Props) {
  const list = limit ? cells.slice(0, limit) : cells;

  if (cells.length === 0) {
    return (
      <p className="text-sm text-gray-400 text-center py-6">未检测到细胞</p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="table-base">
        <thead>
          <tr>
            <th>#</th>
            <th>类别</th>
            <th>置信度</th>
            <th>不确定性</th>
            <th>核质比</th>
            <th>形态解读</th>
          </tr>
        </thead>
        <tbody>
          {list.map((cell, i) => (
            <tr key={i}>
              <td className="text-gray-400">{i + 1}</td>
              <td>
                <span className="font-medium text-gray-800">{classLabel(cell.final_class)}</span>
                <span className="ml-2 text-[11px] text-gray-400">
                  {cell.model_used === 'multiclass' ? '多分类' : '二分类'}
                </span>
              </td>
              <td className="font-semibold text-primary-700">{pct(cell.final_confidence)}</td>
              <td>
                <UncertaintyBadge value={cell.uncertainty} showLabel={false} />
              </td>
              <td>{cell.morphology.核细胞质比}</td>
              <td className="text-gray-500 max-w-xs truncate">{cell.morphology.解释}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}