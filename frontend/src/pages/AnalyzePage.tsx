import { useRef, useState } from 'react';
import { ScanSearch, FileArchive, Images, Play, X, ChevronDown } from 'lucide-react';
import { analyzeBatch, analyzeImage, analyzeZip } from '../libts/api';
import { addHistory } from '../libts/storage';
import { calcRisk, classLabel, pct } from '../libts/format';
import type { CVResult } from '../types';
import StatCard from '../components/ui/StatCard';
import RiskBadge from '../components/ui/RiskBadge';
import CellTable from '../components/ui/CellTable';
import UncertaintyBadge from '../components/ui/UncertaintyBadge';
import Spinner from '../components/ui/Spinner';
import ProbabilityBar from '../components/ui/ProbabilityBar';
import { useToast } from '../components/ui/Toast';
import { AlertTriangle as AlertIcon, Gauge as GaugeIcon, Tag as TagIcon } from 'lucide-react';

type Mode = 'single' | 'batch' | 'zip';

export default function AnalyzePage() {
  const [mode, setMode] = useState<Mode>('single');
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<Record<string, CVResult> | null>(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  const switchMode = (m: Mode) => {
    setMode(m);
    setFiles([]);
    setResults(null);
  };

  const pickFiles = (list: FileList | null) => {
    if (!list) return;
    setFiles(Array.from(list));
    setResults(null);
  };

  const run = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setResults(null);
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
      // 保存历史
      const first = Object.values(out)[0];
      if (first && first.total_cells > 0) {
        const risk = calcRisk(
          first.max_uncertainty,
          first.total_cells ? first.abnormal_cells / first.total_cells : 0
        );
        addHistory({
          id: `${Date.now()}`,
          timestamp: Date.now(),
          filename: mode === 'batch' ? `批量(${files.length}张)` : files[0].name,
          mode: 'quick',
          riskLevel: risk,
          totalCells: first.total_cells,
          abnormalCells: first.abnormal_cells,
        });
      }
      toast.success('分析完成');
    } catch (e) {
      toast.error(`分析失败：${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* 模式切换 */}
      <div className="flex gap-2 flex-wrap">
        {(
          [
            { m: 'single' as Mode, label: '单张分析', icon: Images },
            { m: 'batch' as Mode, label: '批量分析', icon: ScanSearch },
            { m: 'zip' as Mode, label: 'ZIP 批量', icon: FileArchive },
          ]
        ).map(({ m, label, icon: Icon }) => (
          <button
            key={m}
            className={`btn ${mode === m ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => switchMode(m)}
          >
            <Icon size={16} /> {label}
          </button>
        ))}
      </div>

      {/* 文件选择 */}
      <div className="card p-6">
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          multiple={mode === 'batch'}
          accept={mode === 'zip' ? '.zip' : 'image/*'}
          onChange={(e) => pickFiles(e.target.files)}
        />
        <div
          className="border-2 border-dashed border-gray-200 rounded-xl p-8 text-center cursor-pointer hover:border-primary-400 hover:bg-primary-50/40 transition-colors"
          onClick={() => inputRef.current?.click()}
        >
          <div className="text-4xl mb-2">{mode === 'zip' ? '🗜️' : '🖼️'}</div>
          <p className="font-medium text-gray-700">
            {mode === 'single' && '选择单张细胞图像'}
            {mode === 'batch' && '选择多张细胞图像（可多选）'}
            {mode === 'zip' && '选择包含图像的 ZIP 压缩包'}
          </p>
          <p className="text-sm text-gray-400 mt-1">
            {mode === 'zip' ? '自动解压并分析其中所有图片' : '支持 JPG / PNG / BMP，最大 30MB'}
          </p>
          {files.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2 justify-center">
              {files.map((f) => (
                <span key={f.name} className="badge bg-primary-50 text-primary-700 px-3 py-1.5">
                  {f.name}
                  <button
                    className="ml-1 text-primary-400 hover:text-primary-700"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFiles((prev) => prev.filter((x) => x.name !== f.name));
                    }}
                  >
                    <X size={13} />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="mt-4 flex justify-end">
          <button className="btn-primary" disabled={files.length === 0 || loading} onClick={run}>
            {loading ? <Spinner size={16} /> : <Play size={16} />}
            {loading ? '分析中...' : '开始分析'}
          </button>
        </div>
      </div>

      {/* 结果 */}
      {loading && <div className="card p-10 flex justify-center"><Spinner label="模型推理中，请稍候..." /></div>}

      {results && !loading && (
        <div className="space-y-5">
          {Object.entries(results).map(([name, r]) => (
            <ResultCard key={name} name={name} result={r} />
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------- 单图结果卡片 ---------- */
function ResultCard({ name, result }: { name: string; result: CVResult }) {
  const [open, setOpen] = useState(false);
  const risk = calcRisk(
    result.max_uncertainty,
    result.total_cells ? result.abnormal_cells / result.total_cells : 0
  );

  if (result.error) {
    return (
      <div className="card p-5 border-danger-200">
        <div className="flex items-center justify-between">
          <p className="font-medium text-gray-800 truncate">{name}</p>
          <span className="badge bg-danger-100 text-danger-700">分析失败</span>
        </div>
        <p className="text-sm text-danger-600 mt-2">{result.error}</p>
      </div>
    );
  }

  return (
    <div className="card p-5 animate-fade-in">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <p className="font-medium text-gray-800 truncate">{name}</p>
          <RiskBadge level={risk} />
        </div>
        <button
          className="btn-ghost text-xs shrink-0"
          onClick={() => setOpen((v) => !v)}
        >
          <ChevronDown size={15} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
          {open ? '收起' : '展开详情'}
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
        <StatCard icon={Images} label="检测细胞数" value={result.total_cells} tone="primary" />
        <StatCard icon={AlertIcon} label="异常细胞" value={result.abnormal_cells} tone="danger" />
        <StatCard icon={GaugeIcon} label="最大不确定性" value={pct(result.max_uncertainty)} tone="warning" />
        <StatCard icon={TagIcon} label="最高风险类别" value={classLabel(result.highest_risk)} tone="ai" />
      </div>

      {open && (
        <div className="mt-5 space-y-5">
          <CellTable cells={result.cells} />
          {result.cells.length > 0 && (
            <div className="space-y-3">
              <p className="text-sm font-medium text-gray-700">各类别置信度</p>
              <ProbabilitySection cells={result.cells} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* 每个细胞展示其概率分布 */
function ProbabilitySection({ cells }: { cells: CVResult['cells'] }) {
  return (
    <div className="space-y-2">
      {cells.slice(0, 5).map((cell, i) => (
        <details key={i} className="rounded-lg bg-gray-50 p-3">
          <summary className="cursor-pointer text-sm text-gray-700 flex items-center gap-2">
            <span>细胞 #{i + 1}</span>
            <span className="badge bg-primary-50 text-primary-700">{classLabel(cell.final_class)}</span>
            <span className="ml-auto flex items-center gap-2 text-xs text-gray-400">
              <UncertaintyBadge value={cell.uncertainty} showLabel={false} />
            </span>
          </summary>
          <div className="mt-3 space-y-1.5">
            {Object.entries(cell.probabilities).map(([k, v]) => (
              <ProbabilityBar
                key={k}
                label={classLabel(k)}
                value={v}
                colorClass={k === cell.final_class ? 'bg-primary-500' : 'bg-gray-300'}
              />
            ))}
          </div>
        </details>
      ))}
    </div>
  );
}