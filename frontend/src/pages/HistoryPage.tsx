import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { History, RotateCcw, Search, Trash2 } from 'lucide-react';
import { addHistory, clearHistory, getHistory, removeHistory } from '../libts/storage';
import { formatTime } from '../libts/format';
import type { HistoryRecord, RiskLevel } from '../types';
import RiskBadge from '../components/ui/RiskBadge';
import RiskScopeTag from '../components/ui/RiskScopeTag';
import EmptyState from '../components/ui/EmptyState';
import Modal from '../components/ui/Modal';
import { useToast } from '../hooks/useToast';

type RiskFilter = 'all' | 'high' | 'unreadable';
type RangeFilter = 'all' | 'today' | 'week';

const RISK_OPTIONS: { key: RiskFilter; label: string }[] = [
  { key: 'all', label: '全部风险' },
  { key: 'high', label: '高风险' },
  { key: 'unreadable', label: '无法判读' },
];

const RANGE_OPTIONS: { key: RangeFilter; label: string }[] = [
  { key: 'all', label: '全部时间' },
  { key: 'today', label: '今天' },
  { key: 'week', label: '近 7 天' },
];

const isHigh = (r: RiskLevel) => r === 'high' || r === 'emergency';

function startOfToday(): number {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

export default function HistoryPage() {
  const [records, setRecords] = useState<HistoryRecord[]>(() => getHistory());
  const [detail, setDetail] = useState<HistoryRecord | null>(null);
  const [query, setQuery] = useState('');
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all');
  const [rangeFilter, setRangeFilter] = useState<RangeFilter>('all');
  const toast = useToast();

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const since =
      rangeFilter === 'today'
        ? startOfToday()
        : rangeFilter === 'week'
        ? Date.now() - 7 * 24 * 3600 * 1000
        : 0;

    return records.filter((r) => {
      if (r.timestamp < since) return false;
      if (riskFilter === 'high' && !isHigh(r.riskLevel)) return false;
      if (riskFilter === 'unreadable' && r.riskLevel !== 'unreadable') return false;
      if (!q) return true;
      return (
        r.filename.toLowerCase().includes(q) ||
        (r.sessionId ?? '').toLowerCase().includes(q)
      );
    });
  }, [records, query, riskFilter, rangeFilter]);

  /**
   * 删除后提供「撤销」。撤销走 addHistory：它会重新读取 localStorage，
   * 因此即便在撤销前又删了别的记录，也不会把那些记录一起复活。
   */
  const del = (rec: HistoryRecord) => {
    setRecords(removeHistory(rec.id));
    toast.info('已删除该记录', {
      action: {
        label: '撤销',
        onClick: () => setRecords(addHistory(rec)),
      },
    });
  };

  const clearAll = () => {
    if (!window.confirm('确定清空全部本机记录？此操作不可撤销。')) return;
    clearHistory();
    setRecords([]);
    toast.info('已清空全部本机记录');
  };

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="title-serif text-[22px] leading-tight">本机最近记录</h2>
          <p className="mt-1.5 text-[12.5px] text-muted">
            仅保存在当前浏览器，清除缓存或更换设备后不可恢复。最多保留 50 条。
          </p>
        </div>
        {records.length > 0 && (
          <button className="btn-ghost text-sm" onClick={clearAll}>
            <RotateCcw size={15} /> 清空
          </button>
        )}
      </div>

      {/* 工具栏 */}
      {records.length > 0 && (
        <div className="card flex flex-wrap items-center gap-2 p-3">
          <label className="relative flex-1 min-w-[200px]">
            <span className="sr-only">搜索文件名或会话 ID</span>
            <Search
              size={15}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted/70"
              aria-hidden="true"
            />
            <input
              className="input pl-9"
              placeholder="搜索文件名或会话 ID"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>

          <select
            className="input w-auto"
            value={riskFilter}
            onChange={(e) => setRiskFilter(e.target.value as RiskFilter)}
            aria-label="按风险筛选"
          >
            {RISK_OPTIONS.map((o) => (
              <option key={o.key} value={o.key}>
                {o.label}
              </option>
            ))}
          </select>

          <select
            className="input w-auto"
            value={rangeFilter}
            onChange={(e) => setRangeFilter(e.target.value as RangeFilter)}
            aria-label="按时间筛选"
          >
            {RANGE_OPTIONS.map((o) => (
              <option key={o.key} value={o.key}>
                {o.label}
              </option>
            ))}
          </select>

          <span className="ml-auto text-[12px] text-muted tabular">
            {filtered.length} / {records.length} 条
          </span>
        </div>
      )}

      {records.length === 0 ? (
        <div className="card">
          <EmptyState
            icon={History}
            title="暂无诊断记录"
            description="智能诊断或快速分析完成后，会自动在此保存摘要记录"
            action={
              <a href="/diagnose" className="btn-primary">
                开始判读
              </a>
            }
          />
        </div>
      ) : filtered.length === 0 ? (
        <p className="card px-5 py-10 text-center text-[13px] text-muted">
          当前筛选条件下没有记录。
        </p>
      ) : (
        <section className="card">
          <table className="table-base">
            <caption className="sr-only">
              本机诊断记录，按时间倒序排列
            </caption>
            <thead>
              <tr>
                <th scope="col">文件 / 模式</th>
                <th scope="col" className="w-40">时间</th>
                <th scope="col" className="w-16 text-right">细胞</th>
                <th scope="col" className="w-16 text-right">异常</th>
                <th scope="col" className="w-40">风险 / 口径</th>
                <th scope="col" className="w-24 text-right">操作</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.id}>
                  <td className="max-w-[280px]">
                    <button
                      className="block max-w-full truncate text-left text-ink hover:underline"
                      onClick={() => setDetail(r)}
                      title={r.filename}
                    >
                      {r.filename}
                    </button>
                    <span className="text-[11.5px] text-muted">
                      {r.mode === 'agent' ? '智能诊断' : '快速分析'}
                      {r.sessionId && (
                        <span className="tabular"> · {r.sessionId.slice(0, 8)}</span>
                      )}
                    </span>
                  </td>
                  <td className="text-muted tabular whitespace-nowrap">
                    {formatTime(r.timestamp)}
                  </td>
                  <td className="text-right tabular">{r.totalCells ?? '—'}</td>
                  <td className="text-right tabular">{r.abnormalCells ?? '—'}</td>
                  <td>
                    <span className="flex flex-wrap items-center gap-1.5">
                      <RiskBadge level={r.riskLevel} />
                      <RiskScopeTag scope={r.riskScope} level={r.riskLevel} />
                    </span>
                  </td>
                  <td className="text-right">
                    <span className="inline-flex items-center gap-0.5">
                      {r.mode === 'agent' && r.sessionId && (
                        <Link
                          to={`/diagnose?session=${r.sessionId}`}
                          className="px-1.5 py-1 text-[12.5px] text-hematoxylin hover:underline"
                        >
                          打开报告
                        </Link>
                      )}
                      <button
                        className="px-1.5 py-1 text-[12.5px] text-muted hover:underline"
                        onClick={() => setDetail(r)}
                      >
                        摘要
                      </button>
                      <button
                        className="px-1.5 py-1 text-muted hover:text-eosin"
                        onClick={() => del(r)}
                        aria-label={`删除 ${r.filename}`}
                      >
                        <Trash2 size={14} />
                      </button>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <Modal open={!!detail} onClose={() => setDetail(null)} title="诊断记录摘要">
        {detail && (
          <div className="space-y-4 text-[13px]">
            <div className="grid grid-cols-2 gap-3">
              <InfoItem label="文件名" value={detail.filename} />
              <InfoItem label="模式" value={detail.mode === 'agent' ? '智能诊断' : '快速分析'} />
              <InfoItem label="时间" value={formatTime(detail.timestamp)} />
              <div>
                <p className="text-[11.5px] text-muted">风险等级</p>
                <div className="mt-1 flex items-center gap-1.5">
                  <RiskBadge level={detail.riskLevel} size="md" />
                  <RiskScopeTag scope={detail.riskScope} level={detail.riskLevel} />
                </div>
              </div>
              {detail.totalCells != null && (
                <InfoItem label="检测细胞数" value={String(detail.totalCells)} />
              )}
              {detail.abnormalCells != null && (
                <InfoItem label="异常细胞数" value={String(detail.abnormalCells)} />
              )}
              {detail.sessionId && <InfoItem label="会话 ID" value={detail.sessionId} />}
            </div>

            {detail.reportPreview ? (
              <div>
                <p className="text-[11.5px] text-muted mb-2">报告摘要</p>
                <p className="rounded-md border border-hairline bg-glass/60 p-4 leading-relaxed whitespace-pre-wrap text-ink">
                  {detail.reportPreview}
                  {detail.reportPreview.length >= 200 && '…'}
                </p>
                <p className="mt-2 text-[12px] text-muted">
                  本机仅保存摘要。完整报告需在生成时通过「打印 / 存为 PDF」留存。
                </p>
              </div>
            ) : (
              <p className="text-[12.5px] text-muted">
                该记录未保存报告摘要（快速分析只保存细胞计数与风险等级）。
              </p>
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
      <p className="text-[11.5px] text-muted">{label}</p>
      <p className="mt-0.5 truncate font-medium text-ink">{value}</p>
    </div>
  );
}