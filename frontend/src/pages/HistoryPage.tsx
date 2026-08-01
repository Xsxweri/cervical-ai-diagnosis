import { useState } from 'react';
import { Trash2, Eye, RotateCcw, History } from 'lucide-react';
import { getHistory, removeHistory, clearHistory } from '../libts/storage';
import { formatTime } from '../libts/format';
import type { HistoryRecord } from '../types';
import RiskBadge from '../components/ui/RiskBadge';
import EmptyState from '../components/ui/EmptyState';
import Modal from '../components/ui/Modal';
import { useToast } from '../components/ui/Toast';

export default function HistoryPage() {
  const [records, setRecords] = useState<HistoryRecord[]>(() => getHistory());
  const [detail, setDetail] = useState<HistoryRecord | null>(null);
  const toast = useToast();

  const del = (id: string) => {
    setRecords(removeHistory(id));
    toast.info('已删除该记录');
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">诊断历史</h2>
        {records.length > 0 && (
          <button
            className="btn-ghost text-sm"
            onClick={() => {
              clearHistory();
              setRecords([]);
              toast.info('已清空全部记录');
            }}
          >
            <RotateCcw size={15} /> 清空
          </button>
        )}
      </div>

      {records.length === 0 ? (
        <div className="card">
          <EmptyState
            icon={History}
            title="暂无诊断记录"
            description="智能诊断或快速分析完成后，会自动在此保存摘要记录"
          />
        </div>
      ) : (
        <div className="card divide-y divide-gray-50">
          {records.map((r) => (
            <div key={r.id} className="flex items-center gap-4 px-5 py-4 hover:bg-gray-50/60 transition-colors">
              <div className="flex-1 min-w-0">
                <p className="font-medium text-gray-800 truncate">{r.filename}</p>
                <p className="text-xs text-gray-400 mt-0.5">
                  {formatTime(r.timestamp)} · {r.mode === 'agent' ? '智能诊断' : '快速分析'}
                  {r.sessionId && ` · ${r.sessionId.slice(0, 8)}`}
                </p>
              </div>
              <div className="hidden sm:flex gap-4 text-sm text-gray-500">
                {r.totalCells != null && <span>细胞 {r.totalCells}</span>}
                {r.abnormalCells != null && (
                  <span className="text-danger-600">异常 {r.abnormalCells}</span>
                )}
              </div>
              <RiskBadge level={r.riskLevel} />
              <div className="flex gap-1">
                <button className="btn-ghost px-2" onClick={() => setDetail(r)} aria-label="查看详情">
                  <Eye size={16} />
                </button>
                <button className="btn-ghost px-2 text-danger-500 hover:text-danger-700" onClick={() => del(r.id)} aria-label="删除">
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <Modal open={!!detail} onClose={() => setDetail(null)} title="诊断记录详情">
        {detail && (
          <div className="space-y-4 text-sm">
            <div className="grid grid-cols-2 gap-3">
              <InfoItem label="文件名" value={detail.filename} />
              <InfoItem label="模式" value={detail.mode === 'agent' ? '智能诊断' : '快速分析'} />
              <InfoItem label="时间" value={formatTime(detail.timestamp)} />
              <div>
                <p className="text-gray-400 text-xs">风险等级</p>
                <RiskBadge level={detail.riskLevel} size="md" />
              </div>
              {detail.totalCells != null && <InfoItem label="检测细胞数" value={String(detail.totalCells)} />}
              {detail.abnormalCells != null && <InfoItem label="异常细胞数" value={String(detail.abnormalCells)} />}
              {detail.sessionId && <InfoItem label="会话 ID" value={detail.sessionId} />}
            </div>
            {detail.reportPreview && (
              <div>
                <p className="text-gray-400 text-xs mb-2">报告摘要</p>
                <p className="bg-gray-50 rounded-lg p-4 text-gray-700 leading-relaxed whitespace-pre-wrap">
                  {detail.reportPreview}...
                </p>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-gray-400 text-xs">{label}</p>
      <p className="font-medium text-gray-800 mt-0.5 truncate">{value}</p>
    </div>
  );
}