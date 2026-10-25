import { useMemo, useState } from 'react';
import { useRef } from 'react';
import type { ReactNode } from 'react';
import {
  ArrowUpRight,
  FileArchive,
  Images,
  Play,
  ScanSearch,
  X,
} from 'lucide-react';
import { analyzeBatch, analyzeImage, analyzeZip } from '../libts/api';
import { addHistory } from '../libts/storage';
import { RISK_ORDER, calcRisk, classLabel, maxRisk, pct } from '../libts/format';
import type { UncertaintyThresholds } from '../libts/format';
import type { CVResult, RiskLevel } from '../types';
import SpecimenView from '../components/ui/SpecimenView';
import { downloadText, safeFilename, toCsv } from '../libts/csv';
import RiskBadge from '../components/ui/RiskBadge';
import RiskScopeTag from '../components/ui/RiskScopeTag';
import CellTable from '../components/ui/CellTable';
import UncertaintyBadge from '../components/ui/UncertaintyBadge';
import Spinner from '../components/ui/Spinner';
import ProbabilityBar from '../components/ui/ProbabilityBar';
import Drawer from '../components/ui/Drawer';
import { useToast } from '../hooks/useToast';

type Mode = 'single' | 'batch' | 'zip';
type Filter = 'all' | 'high' | 'unreadable';

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: '全部' },
  { key: 'high', label: '高风险' },
  { key: 'unreadable', label: '无法判读' },
];

const MODES: { m: Mode; label: string; icon: typeof Images }[] = [
  { m: 'single', label: '单张分析', icon: Images },
  { m: 'batch', label: '批量分析', icon: ScanSearch },
  { m: 'zip', label: 'ZIP 批量', icon: FileArchive },
];

/**
 * 风险判定：优先信任后端的 risk_level；
 * 但只要"未检出细胞/无法判读"就绝不允许落到"低风险"。
 */
function pickRisk(result: CVResult): RiskLevel {
  if (result.error) return 'unreadable';
  if (result.interpretable === false) return 'unreadable';
  if (result.risk_level) return result.risk_level;
  if (!result.total_cells) return 'unreadable';
  const ratio = result.abnormal_cells / result.total_cells;
  return calcRisk(result.max_uncertainty, ratio);
}

function isReadable(result: CVResult): boolean {
  return !result.error && result.interpretable !== false && (result.total_cells ?? 0) > 0;
}

const isHigh = (r: RiskLevel) => r === 'high' || r === 'emergency';

/** 排序用：无法判读排在最后，不参与风险比较 */
const riskRank = (r: RiskLevel) => (r === 'unreadable' ? -1 : RISK_ORDER.indexOf(r));

interface Row {
  name: string;
  result: CVResult;
  risk: RiskLevel;
}

export default function AnalyzePage() {
  const [mode, setMode] = useState<Mode>('single');
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<Record<string, CVResult> | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<Filter>('all');
  const [activeName, setActiveName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  /** 结果按文件名回查原始 File（ZIP 模式下没有图片字节，只留占位） */
  const fileMap = useMemo(() => {
    const map = new Map<string, File>();
    if (mode === 'zip') return map;
    files.forEach((f) => {
      if (!map.has(f.name)) map.set(f.name, f);
    });
    return map;
  }, [files, mode]);

  const rows = useMemo<Row[]>(() => {
    if (!results) return [];
    return Object.entries(results)
      .map(([name, result]) => ({ name, result, risk: pickRisk(result) }))
      .sort((a, b) => {
        const byRisk = riskRank(b.risk) - riskRank(a.risk);
        if (byRisk !== 0) return byRisk;
        return (b.result.max_uncertainty ?? 0) - (a.result.max_uncertainty ?? 0);
      });
  }, [results]);

  const summary = useMemo(() => {
    const total = rows.length;
    const readable = rows.filter((r) => isReadable(r.result)).length;
    const high = rows.filter((r) => isReadable(r.result) && isHigh(r.risk)).length;
    const unreadable = rows.filter((r) => r.risk === 'unreadable').length;
    return { total, readable, high, unreadable };
  }, [rows]);

  const visible = rows.filter((r) => {
    if (filter === 'high') return isReadable(r.result) && isHigh(r.risk);
    if (filter === 'unreadable') return r.risk === 'unreadable';
    return true;
  });

  const activeRow = rows.find((r) => r.name === activeName) ?? null;

  const switchMode = (m: Mode) => {
    setMode(m);
    setFiles([]);
    setResults(null);
    setFilter('all');
    setActiveName(null);
  };

  const pickFiles = (list: FileList | null) => {
    if (!list) return;
    setFiles(Array.from(list));
    setResults(null);
    setActiveName(null);
  };

  const exportCsv = () => {
    if (rows.length === 0) return;
    const csv = toCsv(
      ['文件名', '风险等级', '风险口径', '细胞数', '异常细胞数', '不确定性', '最高风险类别', '错误'],
      rows.map(({ name, result, risk }) => [
        name,
        risk,
        result.risk_scope ?? '',
        result.total_cells ?? '',
        result.abnormal_cells ?? '',
        result.max_uncertainty ?? '',
        result.highest_risk ?? '',
        result.error ?? '',
      ]),
    );
    const stamp = new Date().toISOString().slice(0, 10);
    downloadText(`${safeFilename(`快速分析_${stamp}`, '快速分析')}.csv`, csv);
  };

  const run = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setResults(null);
    setActiveName(null);
    try {
      let out: Record<string, CVResult> = {};
      if (mode === 'single') {
        const r = await analyzeImage(files[0]);
        out = { [files[0].name]: r };
      } else if (mode === 'batch') {
        out = await analyzeBatch(files);
      } else {
        out = await analyzeZip(files[0]);
      }
      setResults(out);

      // 历史记录必须是整批汇总，不能只取第一张（否则单张结果会被误读为整批结论）
      const entries = Object.values(out);
      const readable = entries.filter(isReadable);
      if (readable.length > 0) {
        const totalCells = readable.reduce((s, r) => s + (r.total_cells ?? 0), 0);
        const abnormalCells = readable.reduce((s, r) => s + (r.abnormal_cells ?? 0), 0);
        addHistory({
          id: `${Date.now()}`,
          timestamp: Date.now(),
          filename:
            mode === 'single'
              ? files[0].name
              : `批量(${readable.length}/${entries.length}张)`,
          mode: 'quick',
          riskLevel: maxRisk(readable.map(pickRisk)),
          riskScope: readable.some((r) => r.risk_scope === 'case') ? 'case' : 'cell',
          totalCells,
          abnormalCells,
          hasConflict:
            new Set(
              readable.flatMap((r) =>
                r.cells
                  .filter((c) => c.binary_result.class === 'abnormal')
                  .map((c) => c.final_class),
              ),
            ).size > 1,
        });
      } else if (entries.length > 0) {
        toast.info('所选图像均无法判读，未写入历史');
      }
      toast.success('分析完成');
    } catch (e) {
      toast.error(`分析失败：${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* 模式切换 + 上传 */}
      <div className="card p-5">
        <div className="flex gap-2 flex-wrap">
          {MODES.map(({ m, label, icon: Icon }) => (
            <button
              key={m}
              className={`btn ${mode === m ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => switchMode(m)}
              aria-pressed={mode === m}
            >
              <Icon size={16} /> {label}
            </button>
          ))}
        </div>

        <input
          ref={inputRef}
          type="file"
          className="hidden"
          multiple={mode === 'batch'}
          accept={mode === 'zip' ? '.zip' : 'image/*'}
          onChange={(e) => pickFiles(e.target.files)}
        />

        <div
          className="mt-4 rounded-md border border-dashed border-hairline bg-glass/50 p-6 text-center cursor-pointer transition-colors hover:border-muted/50 hover:bg-glass"
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
          aria-label="选择要分析的图像文件"
        >
          <FileArchive size={20} className="mx-auto mb-2 text-muted/70" aria-hidden="true" />
          <p className="text-[13.5px] text-ink">
            {mode === 'single' && '选择单张细胞图像'}
            {mode === 'batch' && '选择多张细胞图像（最多 10 张）'}
            {mode === 'zip' && '选择包含图像的 ZIP 压缩包（最多 20 张）'}
          </p>
          <p className="mt-1 text-[12px] text-muted">
            {mode === 'zip' ? '自动解压并分析其中所有图片' : '支持 JPG / PNG / BMP，单文件最大 30MB'}
          </p>
        </div>

        {files.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {files.map((f, idx) => (
              <span
                key={`${f.name}-${idx}`}
                className="badge bg-primary-50 text-hematoxylin px-3 py-1.5"
              >
                <span className="max-w-[240px] truncate">{f.name}</span>
                <button
                  className="ml-1 text-primary-400 hover:text-hematoxylin"
                  onClick={() => setFiles((prev) => prev.filter((_, i) => i !== idx))}
                  aria-label={`移除 ${f.name}`}
                >
                  <X size={13} />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="mt-4 flex justify-end">
          <button
            className="btn-primary"
            disabled={files.length === 0 || loading}
            onClick={run}
          >
            {loading ? <Spinner size={16} /> : <Play size={16} />}
            {loading ? '分析中…' : '开始分析'}
          </button>
        </div>
      </div>

      {loading && (
        <div className="card flex justify-center p-10">
          <Spinner label="模型推理中，请稍候…" />
        </div>
      )}

      {/* 分诊表 */}
      {!loading && mode !== 'single' && rows.length > 0 && (
        <section className="card">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline px-5 py-3">
            <p className="text-[12.5px] text-muted">
              已分析 <span className="tabular text-ink">{summary.total}</span> 张 ·
              可判读 <span className="tabular text-ink">{summary.readable}</span>
              {summary.high > 0 && (
                <>
                  {' · '}高风险 <span className="tabular text-risk-highInk">{summary.high}</span>
                </>
              )}
              {summary.unreadable > 0 && (
                <>
                  {' · '}无法判读{' '}
                  <span className="tabular text-ink">{summary.unreadable}</span>
                </>
              )}
            </p>
            <div className="flex items-center gap-1">
              {FILTERS.map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setFilter(key)}
                  aria-pressed={filter === key}
                  className={`rounded-md px-2.5 py-1 text-[12.5px] transition-colors ${
                    filter === key
                      ? 'bg-primary-50 font-medium text-hematoxylin'
                      : 'text-muted hover:bg-glass hover:text-ink'
                  }`}
                >
                  {label}
                </button>
              ))}
              <span className="ml-2 text-[12px] text-muted">按风险降序</span>
              <button
                className="btn-ghost ml-2 text-[12.5px]"
                onClick={exportCsv}
                aria-label="导出结果为 CSV"
              >
                导出 CSV
              </button>
            </div>
          </div>

          {visible.length === 0 ? (
            <p className="px-5 py-10 text-center text-[13px] text-muted">
              当前筛选下没有结果。
            </p>
          ) : (
            <table className="table-base">
              <caption className="sr-only">
                快速分析结果，按风险等级与不确定性降序排列
              </caption>
              <thead>
                <tr>
                  <th scope="col" className="w-[92px]">视野</th>
                  <th scope="col">文件</th>
                  <th scope="col" className="w-16 text-right">细胞</th>
                  <th scope="col" className="w-16 text-right">异常</th>
                  <th scope="col" className="w-44">风险 / 口径</th>
                  <th scope="col" className="w-24 text-right">不确定性</th>
                  <th scope="col" className="w-16 text-right">操作</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((row) => {
                  const file = fileMap.get(row.name);
                  const unreadable = row.risk === 'unreadable';
                  return (
                    <tr
                      key={row.name}
                      tabIndex={0}
                      onClick={() => setActiveName(row.name)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          setActiveName(row.name);
                        }
                      }}
                      className={`cursor-pointer ${unreadable ? 'bg-risk-unreadableBg/50' : ''}`}
                    >
                      <td className="py-2">
                        <SpecimenView
                          file={file}
                          cells={row.result.cells}
                          alt=""
                          className="h-14 w-16 rounded border border-hairline"
                        />
                      </td>
                      <td className="max-w-[240px]">
                        <span className="block truncate text-ink" title={row.name}>
                          {row.name}
                        </span>
                        {row.result.error && (
                          <span className="text-[11.5px] text-risk-highInk">
                            分析失败
                          </span>
                        )}
                        {!row.result.error && unreadable && (
                          <span className="text-[11.5px] text-muted">未检出可判读细胞</span>
                        )}
                      </td>
                      <td className="text-right tabular">
                        {unreadable ? '—' : (row.result.total_cells ?? '—')}
                      </td>
                      <td className="text-right tabular">
                        {unreadable ? '—' : (row.result.abnormal_cells ?? '—')}
                      </td>
                      <td>
                        <span className="flex flex-wrap items-center gap-1.5">
                          <RiskBadge level={row.risk} />
                          <RiskScopeTag scope={row.result.risk_scope} level={row.risk} />
                        </span>
                      </td>
                      <td className="text-right">
                        {unreadable ? (
                          <span className="text-muted">—</span>
                        ) : (
                          <UncertaintyBadge
                            value={row.result.max_uncertainty}
                            showLabel={false}
                            thresholds={row.result.uncertainty_thresholds}
                          />
                        )}
                      </td>
                      <td className="text-right">
                        <button
                          className="inline-flex items-center gap-0.5 text-[12.5px] text-hematoxylin hover:underline"
                          onClick={(e) => {
                            e.stopPropagation();
                            setActiveName(row.name);
                          }}
                          aria-label={`展开 ${row.name} 详情`}
                        >
                          展开
                          <ArrowUpRight size={13} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>
      )}

      {/* 单张模式：沿用同一套详情布局，直接内联，不另起 UI */}
      {!loading && mode === 'single' && rows.length === 1 && (
        <section className="card p-5">
          <ResultDetail
            result={rows[0].result}
            file={fileMap.get(rows[0].name)}
          />
        </section>
      )}

      {/* 批量 / ZIP：详情走 Drawer */}
      <Drawer
        open={mode !== 'single' && !!activeRow}
        onClose={() => setActiveName(null)}
        title={activeRow?.name}
        subtitle={
          activeRow ? (
            <span className="flex flex-wrap items-center gap-1.5">
              <RiskBadge level={activeRow.risk} />
              <RiskScopeTag scope={activeRow.result.risk_scope} level={activeRow.risk} />
            </span>
          ) : undefined
        }
      >
        {activeRow && (
          <ResultDetail
            result={activeRow.result}
            file={fileMap.get(activeRow.name)}
          />
        )}
      </Drawer>
    </div>
  );
}


/* ---------- 指标条（替代卡片套卡片） ---------- */
function MetricStrip({ items }: { items: { label: string; value: ReactNode }[] }) {
  return (
    <dl className="flex flex-wrap gap-x-6 gap-y-2 border-y border-hairline py-3">
      {items.map((it) => (
        <div key={it.label} className="flex items-baseline gap-2">
          <dt className="text-[11.5px] text-muted">{it.label}</dt>
          <dd className="tabular text-[13px] text-ink">{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/* ---------- 单图详情（Drawer 与单张模式共用） ---------- */
function ResultDetail({ result, file }: { result: CVResult; file?: File }) {
  const risk = pickRisk(result);
  const [showAll, setShowAll] = useState(false);
  const unreadable = risk === 'unreadable';

  return (
    <div className="space-y-5">
      <SpecimenView
        file={file}
        cells={result.cells}
        alt="细胞学视野与检测框"
        className="h-64 w-full rounded-md border border-hairline"
      />

      {result.error && (
        <div className="border-l-2 border-eosin pl-3 text-[13px] leading-relaxed text-eosin">
          {result.error}
        </div>
      )}

      {!result.error && unreadable && (
        <div className="rounded-md border-l-2 border-risk-medium bg-risk-mediumBg px-3 py-2 text-[13px] leading-relaxed text-risk-mediumInk">
          <p className="font-medium">无法判读</p>
          <p className="mt-1">
            {result.unreadable_reason ?? '未检测到可判读的宫颈细胞，本图无法给出风险分级。'}
          </p>
          <p className="mt-1 text-[12px]">
            请确认图像为细胞学视野、清晰度足够，且未为大面积空白或非细胞内容。
          </p>
        </div>
      )}

      {!unreadable && (
        <>
          <MetricStrip
            items={[
              { label: '检测细胞', value: result.total_cells },
              { label: '异常细胞', value: result.abnormal_cells },
              {
                label: `不确定性 (P${result.uncertainty_aggregation?.replace('p', '') ?? '90'})`,
                value: pct(result.max_uncertainty),
              },
              { label: '最高风险类别', value: classLabel(result.highest_risk) },
            ]}
          />

          <CellTable
            cells={result.cells}
            limit={showAll ? undefined : 20}
            thresholds={result.uncertainty_thresholds}
          />

          {result.cells.length > 20 && (
            <button
              className="btn-ghost text-[12.5px]"
              onClick={() => setShowAll((v) => !v)}
            >
              {showAll
                ? '仅显示前 20 个细胞'
                : `显示全部 ${result.cells.length} 个细胞`}
            </button>
          )}

          {result.cells.length > 0 && (
            <ProbabilitySection
              cells={result.cells}
              thresholds={result.uncertainty_thresholds}
            />
          )}
        </>
      )}
    </div>
  );
}

/* ---------- 每个细胞的概率分布 ---------- */
function ProbabilitySection({
  cells,
  thresholds,
}: {
  cells: CVResult['cells'];
  thresholds?: UncertaintyThresholds;
}) {
  return (
    <div className="space-y-2">
      {cells.slice(0, 5).map((cell, i) => (
        <details key={i} className="rounded-md bg-glass/60 p-3">
          <summary className="flex cursor-pointer items-center gap-2 text-[13px] text-ink">
            <span className="tabular">细胞 #{i + 1}</span>
            <span className="badge bg-primary-50 text-hematoxylin">
              {classLabel(cell.final_class)}
            </span>
            <span className="ml-auto flex items-center gap-2">
              <UncertaintyBadge
                value={cell.uncertainty}
                showLabel={false}
                thresholds={thresholds}
              />
            </span>
          </summary>
          <div className="mt-3 space-y-1.5">
            {Object.entries(cell.probabilities).map(([k, v]) => (
              <ProbabilityBar
                key={k}
                label={classLabel(k)}
                value={v}
                colorClass={k === cell.final_class ? 'bg-primary-500' : 'bg-hairline'}
              />
            ))}
          </div>
        </details>
      ))}
    </div>
  );
}