import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, ArrowUpRight, Info, ScanSearch } from 'lucide-react';
import { fetchSystemInfo } from '../libts/api';
import { getHistory } from '../libts/storage';
import { formatTime } from '../libts/format';
import type { RiskLevel, SystemInfoResponse } from '../types';
import RiskBadge from '../components/ui/RiskBadge';
import RiskScopeTag from '../components/ui/RiskScopeTag';
import EmptyState from '../components/ui/EmptyState';

type Filter = 'all' | 'high' | 'unreadable';

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: '全部' },
  { key: 'high', label: '高风险' },
  { key: 'unreadable', label: '无法判读' },
];

const HIGH: RiskLevel[] = ['high', 'emergency'];

function isToday(ts: number): boolean {
  const d = new Date(ts);
  const now = new Date();
  return (
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
  );
}

export default function DashboardPage() {
  const [health, setHealth] = useState<SystemInfoResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [filter, setFilter] = useState<Filter>('all');
  const records = useMemo(() => getHistory(), []);

  useEffect(() => {
    let cancelled = false;
    fetchSystemInfo()
      .then((h) => {
        if (!cancelled) setHealth(h);
      })
      .catch(() => {
        if (!cancelled) setHealthError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const today = records.filter((r) => isToday(r.timestamp));
  const todayHigh = today.filter((r) => HIGH.includes(r.riskLevel)).length;
  const todayUnreadable = today.filter((r) => r.riskLevel === 'unreadable').length;
  const todayConflict = today.filter((r) => r.hasConflict).length;

  const visible = records.filter((r) => {
    if (filter === 'high') return HIGH.includes(r.riskLevel);
    if (filter === 'unreadable') return r.riskLevel === 'unreadable';
    return true;
  });

  return (
    <div className="space-y-6">
      {/* 标题行 */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h2 className="title-serif text-[26px] leading-tight">今日判读</h2>
          <p className="mt-1.5 text-[13px] text-muted">
            {today.length > 0 ? (
              <>
                今日新增 {today.length} 例
                {todayHigh > 0 && <> · 高风险 {todayHigh}</>}
                {todayUnreadable > 0 && <> · 无法判读 {todayUnreadable}</>}
                {todayConflict > 0 && <> · 证据冲突 {todayConflict}</>}
              </>
            ) : (
              '今天还没有判读记录。'
            )}
          </p>
        </div>
        <Link to="/diagnose" className="btn-primary">
          <ScanSearch size={16} /> 开始判读
        </Link>
      </div>

      {/* 病例表 */}
      <section className="card">
        <div className="flex items-center justify-between gap-3 px-5 py-3 border-b border-hairline flex-wrap">
          <h3 className="text-[14px] font-medium text-ink">最近病例</h3>
          <div className="flex items-center gap-1">
            {FILTERS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                aria-pressed={filter === key}
                className={`px-2.5 py-1 rounded-md text-[12.5px] transition-colors ${
                  filter === key
                    ? 'bg-primary-50 text-hematoxylin font-medium'
                    : 'text-muted hover:text-ink hover:bg-glass'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {visible.length === 0 ? (
          <EmptyState
            icon={ScanSearch}
            title={records.length === 0 ? '今天还没有判读记录' : '当前筛选下没有病例'}
            description={
              records.length === 0
                ? '上传一张细胞学图像开始。'
                : '换一个筛选条件试试。'
            }
            action={
              records.length === 0 ? (
                <Link to="/diagnose" className="btn-primary">
                  开始判读
                </Link>
              ) : undefined
            }
          />
        ) : (
          <table className="table-base">
            <caption className="sr-only">最近判读病例，按风险等级与时间排序</caption>
            <thead>
              <tr>
                <th scope="col">文件</th>
                <th scope="col">时间</th>
                <th scope="col" className="text-right">细胞</th>
                <th scope="col" className="text-right">异常</th>
                <th scope="col">风险 / 口径</th>
                <th scope="col" className="text-right">操作</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr key={r.id}>
                  <td className="max-w-[220px]">
                    <span className="block truncate text-ink">{r.filename}</span>
                    <span className="text-[11.5px] text-muted">
                      {r.mode === 'agent' ? '智能诊断' : '快速分析'}
                    </span>
                  </td>
                  <td className="text-muted tabular whitespace-nowrap">
                    {formatTime(r.timestamp)}
                  </td>
                  <td className="text-right tabular">{r.totalCells ?? '—'}</td>
                  <td className="text-right tabular">
                    {r.abnormalCells ?? '—'}
                  </td>
                  <td>
                    <span className="flex items-center gap-1.5 flex-wrap">
                      <RiskBadge level={r.riskLevel} />
                      <RiskScopeTag scope={r.riskScope} level={r.riskLevel} />
                      {r.hasConflict && (
                        <span
                          className="badge bg-risk-mediumBg text-risk-mediumInk"
                          title="异常细胞分型不止一种，建议复核"
                        >
                          证据冲突
                        </span>
                      )}
                    </span>
                  </td>
                  <td className="text-right">
                    <Link
                      to="/history"
                      className="inline-flex items-center gap-0.5 text-[12.5px] text-hematoxylin hover:underline"
                    >
                      查看
                      <ArrowUpRight size={13} />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* 状态带 */}
      <div className="flex items-center gap-3 px-1 text-[12px] text-muted flex-wrap">
        {healthError ? (
          <span className="inline-flex items-center gap-1 text-risk-highInk">
            <AlertTriangle size={14} /> 无法连接判读服务
          </span>
        ) : health ? (
          <span className="tabular">
            模型已加载 · {health.device}
            {!health.rag_available && ' · 知识库不可用（本次报告无文献依据）'}
          </span>
        ) : (
          <span>正在获取系统状态…</span>
        )}

        {health && (
          <details className="ml-auto">
            <summary className="cursor-pointer list-none inline-flex items-center gap-1 hover:text-ink">
              <Info size={13} /> 系统信息
            </summary>
            <dl className="mt-2 rounded-md border border-hairline bg-white p-3 space-y-1 tabular">
              <div className="flex gap-2">
                <dt className="text-muted w-24 shrink-0">检测模型</dt>
                <dd className="truncate">{health.models.yolo}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-muted w-24 shrink-0">二分类</dt>
                <dd className="truncate">{health.models.二分类}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-muted w-24 shrink-0">多分类</dt>
                <dd className="truncate">{health.models.多分类}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-muted w-24 shrink-0">不确定性阈值</dt>
                <dd>
                  中 {health.config.不确定阈值_medium} / 高 {health.config.不确定阈值_high}
                </dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-muted w-24 shrink-0">异常 CONF</dt>
                <dd>{health.config.异常的CONF阈值}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-muted w-24 shrink-0">不确定性采样</dt>
                <dd>MC Dropout × {health.config.mc_dropout_samples}</dd>
              </div>
            </dl>
          </details>
        )}
      </div>
    </div>
  );
}