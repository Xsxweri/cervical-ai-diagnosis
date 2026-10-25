import { useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Send, AlertTriangle, Sparkles, RotateCcw, MessageCircle, X, Printer, Copy } from 'lucide-react';
import { useDiagnosis, type ChatMessage } from '../hooks/useDiagnosis';
import { riskFromText, formatTime, pct, classLabel } from '../libts/format';
import type { CellResult, RiskLevel } from '../types';
import { getDoctor } from '../libts/storage';
import ImageDropzone from '../components/ui/ImageDropzone';
import Spinner from '../components/ui/Spinner';
import RiskBadge from '../components/ui/RiskBadge';
import RiskScopeTag from '../components/ui/RiskScopeTag';
import Drawer from '../components/ui/Drawer';
import { parseReportSections } from '../libts/report';
import SpecimenView from '../components/ui/SpecimenView';
import ReviewTag from '../components/ui/ReviewTag';


const STEPS = ['上传图像', 'AI 分析', '报告', '追问'];

export default function DiagnosePage() {
  const diag = useDiagnosis();
  const [file, setFile] = useState<File | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  useEffect(() => {
    const sid = searchParams.get('session');
    if (!sid) return;
    diag.loadSession(sid);
    // 立即清掉参数：避免刷新页面时重复拉取、也避免 URL 里长期带 session
    setSearchParams({}, { replace: true });
    // 只在挂载时执行一次
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const stepIndex =
    diag.step === 'upload' ? 0
    : diag.step === 'analyzing' || diag.step === 'clarification' || diag.step === 'unreadable' ? 1
    : diag.step === 'report' ? 2
    : 3;

  return (
    <div className="space-y-6">
      {/* 步骤指示器 */}
      <div className="card p-4 flex items-center justify-between max-w-3xl">
        {STEPS.map((label, i) => (
          <div key={label} className="flex items-center flex-1 last:flex-none">
            <div className="flex flex-col items-center gap-1">
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold transition-colors ${
                  i < stepIndex
                    ? 'bg-primary-600 text-white'
                    : i === stepIndex
                    ? 'bg-primary-100 text-primary-700 ring-2 ring-primary-500/50'
                    : 'bg-gray-100 text-gray-400'
                }`}
              >
                {i + 1}
              </div>
              <span className={`text-xs ${i === stepIndex ? 'text-primary-700 font-medium' : 'text-gray-400'}`}>
                {label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={`flex-1 h-0.5 mx-2 rounded ${i < stepIndex ? 'bg-primary-500' : 'bg-gray-200'}`} />
            )}
          </div>
        ))}
      </div>

      {diag.step === 'upload' && (
        <UploadStep file={file} onFileChange={setFile} onStart={() => file && diag.start(file)} />
      )}
      {diag.step === 'analyzing' && <AnalyzingStep file={file} onCancel={diag.cancel} />}
      {diag.step === 'clarification' && (
        <ClarificationStep questions={diag.questions} loading={diag.isLoading} onSubmit={diag.submitClarification} />
      )}
      {diag.step === 'unreadable' && (
        <UnreadableStep
          message={diag.errorMessage ?? '未检测到可判读的宫颈细胞。'}
          onReset={diag.reset}
        />
      )}
      {(diag.step === 'report' || diag.step === 'chat') && (
        <ReportStep
          report={diag.report}
          riskLevel={diag.riskLevel}
          riskScope={diag.riskScope}
          recommendations={diag.recommendations}
          decisionReason={diag.decisionReason}
          clarificationRounds={diag.clarificationRounds}
          reviewRequired={diag.reviewRequired}
          conflictCount={diag.conflictCount}
          sessionId={diag.sessionId}
          reportAt={diag.reportAt}
          metrics={diag.cvMetrics}
          cells={diag.cvCells}
          specimenFile={file}
          specimenUrl={diag.specimenUrl}
          chatOpen={diag.step === 'chat'}
          onChat={diag.startChat}
          onReset={diag.reset}
        />
      )}
      {/* 追问：覆盖在报告之上的抽屉，报告保持可见 */}
      <Drawer
        open={diag.step === 'chat'}
        onClose={diag.closeChat}
        title="医生追问"
        subtitle={
          <span className="text-muted">
            针对本次判读报告提问；报告内容不会随对话改变。
          </span>
        }
        width="max-w-xl"
      >
        <ChatStep
          messages={diag.messages}
          sending={diag.isSending}
          onSend={diag.sendMessage}
        />
      </Drawer>
    </div>
  );
}

/* ---------- 上传步骤 ---------- */
function UploadStep({
  file,
  onFileChange,
  onStart,
}: {
  file: File | null;
  onFileChange: (f: File | null) => void;
  onStart: () => void;
}) {
  return (
    <div className="max-w-2xl mx-auto space-y-5 animate-fade-in">
      {/* 移除图片时必须同步清空父组件的 file，否则会继续上传已删除的图 */}
      <ImageDropzone onFile={(f) => onFileChange(f)} onRemove={() => onFileChange(null)} />
      <div className="flex items-center justify-between gap-3">
        {file ? (
          <p className="text-sm text-gray-500 truncate">
            已选择：<span className="font-medium text-gray-700">{file.name}</span>
            <span className="ml-2 text-gray-400">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
          </p>
        ) : (
          <span />
        )}
        <button className="btn-primary ml-auto" disabled={!file} onClick={onStart}>
          <Sparkles size={16} /> 开始 AI 诊断
        </button>
      </div>
    </div>
  );
}

/* ---------- 分析中（模拟流水线进度） ---------- */
const PHASES = ['识别细胞', '评估风险', '生成报告'];

function AnalyzingStep({
  file,
  onCancel,
}: {
  file?: File | null;
  onCancel: () => void;
}) {
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      // 只前进到最后一个阶段后停住，不循环
      setIdx((i) => (i >= PHASES.length - 1 ? i : i + 1));
    }, 1600);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="mx-auto max-w-xl">
      <div className="card p-5">
        {/* 让医生看到自己上传的图，而不是只有模型名和转圈 */}
        <SpecimenView
          file={file ?? undefined}
          alt="待分析的细胞学图像"
          className="h-44 w-full rounded-md border border-hairline"
        />

        <div className="mt-4 flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-2 text-[13px]">
            {PHASES.map((label, i) => (
              <span key={label} className="flex items-center gap-2">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    i < idx
                      ? 'bg-success-500'
                      : i === idx
                      ? 'bg-hematoxylin animate-pulse-soft'
                      : 'bg-hairline'
                  }`}
                />
                <span className={i <= idx ? 'text-ink' : 'text-muted'}>{label}</span>
                {i < PHASES.length - 1 && <span className="text-hairline">…</span>}
              </span>
            ))}
          </div>
          <button className="btn-secondary shrink-0" onClick={onCancel}>
            <X size={15} /> 取消
          </button>
        </div>

        <p className="mt-3 text-[12px] text-muted">
          阶段提示，非精确进度；耗时取决于图像复杂度与服务器负载。
        </p>
      </div>
    </div>
  );
}

/* ---------- 澄清步骤 ---------- */
const WHY_HINTS: { match: RegExp; hint: string }[] = [
  { match: /年龄|生育|月经/, hint: '年龄与生育史决定细胞学判读的基线风险。' },
  { match: /HPV/i, hint: '高危型 HPV 状态决定分流路径与随访间隔。' },
  { match: /出血|症状/, hint: '接触性出血等症状用于区分炎症修复与上皮内病变。' },
  { match: /治疗|LEEP|锥切|手术|活检/, hint: '既往治疗史会改变局部细胞形态学表现。' },
  { match: /免疫|吸烟|高危因素/, hint: '免疫抑制与吸烟会提高病变进展风险。' },
  { match: /筛查|细胞学|既往/, hint: '既往筛查结果用于判断是新发异常还是持续异常。' },
];

function whyAsked(question: string): string {
  return (
    WHY_HINTS.find((h) => h.match.test(question))?.hint ??
    '补充该项信息有助于提高判读可靠性。'
  );
}

function ClarificationStep({
  questions,
  loading,
  onSubmit,
}: {
  questions: string[];
  loading: boolean;
  onSubmit: (answers: string[]) => void;
}) {
  const [answers, setAnswers] = useState<string[]>(() => questions.map(() => ''));
  const [declined, setDeclined] = useState<boolean[]>(() => questions.map(() => false));

  // 问题集变化时重置，避免上一批问题的回答串到新问题上
  useEffect(() => {
    setAnswers(questions.map(() => ''));
    setDeclined(questions.map(() => false));
  }, [questions]);

  // 勾选「不详」也算已完成，避免因单项未知而无法提交
  const valid = questions.every(
    (_, i) => declined[i] === true || !!(answers[i] ?? '').trim(),
  );

  const submit = () => {
    // 「不详」作为明确内容提交，否则 join('；') 会产生 "；；" 这种空噪声
    onSubmit(
      questions.map((_, i) => (declined[i] ? '不详' : (answers[i] ?? '').trim())),
    );
  };

  return (
    <div className="max-w-2xl mx-auto space-y-5">
      <div className="card p-5 border-warning-200 bg-warning-50/60">
        <div className="flex items-start gap-3">
          <AlertTriangle size={20} className="text-warning-600 mt-0.5 shrink-0" />
          <div>
            <h3 className="font-semibold text-warning-800">需要补充临床信息</h3>
            <p className="text-sm text-warning-700 mt-1">
              AI 判读的不确定性偏高。补充下列信息有助于缩小鉴别范围；
              确实无法提供的项目，勾选「不详」即可继续。
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {questions.map((q, i) => {
          const isDeclined = declined[i] === true;
          return (
            <div key={`${i}-${q}`} className="card p-4">
              <label
                htmlFor={`clarify-${i}`}
                className="block text-sm font-medium text-ink"
              >
                {i + 1}. {q}
              </label>
              <p className="mt-1 mb-2.5 text-[12px] leading-relaxed text-muted">
                {whyAsked(q)}
              </p>
              <textarea
                id={`clarify-${i}`}
                className="input min-h-[84px] resize-y leading-relaxed"
                rows={3}
                placeholder={
                  isDeclined
                    ? '已标记为「不详」'
                    : '可分段描述：年龄、症状与持续时间、既往筛查或治疗情况…'
                }
                value={isDeclined ? '' : (answers[i] ?? '')}
                disabled={isDeclined || loading}
                aria-describedby={`clarify-why-${i}`}
                onChange={(e) => {
                  const next = [...answers];
                  next[i] = e.target.value;
                  setAnswers(next);
                }}
              />
              <span id={`clarify-why-${i}`} className="sr-only">
                {whyAsked(q)}
              </span>
              <label
                htmlFor={`clarify-decline-${i}`}
                className="mt-2.5 inline-flex items-center gap-2 text-[12.5px] text-muted cursor-pointer select-none"
              >
                <input
                  id={`clarify-decline-${i}`}
                  type="checkbox"
                  className="h-4 w-4 rounded border-hairline accent-hematoxylin focus-visible:ring-2 focus-visible:ring-hematoxylin/40"
                  checked={isDeclined}
                  disabled={loading}
                  onChange={(e) => {
                    const next = [...declined];
                    next[i] = e.target.checked;
                    setDeclined(next);
                  }}
                />
                不详 / 暂无法提供
              </label>
            </div>
          );
        })}
      </div>

      <button className="btn-primary w-full" disabled={!valid || loading} onClick={submit}>
        {loading ? (
          <Spinner size={16} />
        ) : (
          <>
            <Sparkles size={16} /> 提交并重新判读
          </>
        )}
      </button>
    </div>
  );
}

/* ---------- 不可判读步骤 ---------- */
function UnreadableStep({ message, onReset }: { message: string; onReset: () => void }) {
  return (
    <div className="max-w-2xl mx-auto space-y-5 animate-fade-in">
      <div className="card p-6 border-warning-200 bg-warning-50/60">
        <div className="flex items-start gap-3">
          <AlertTriangle size={20} className="text-warning-600 mt-0.5 shrink-0" />
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-semibold text-warning-800">无法判读</h3>
              <RiskBadge level="unreadable" />
            </div>
            <p className="text-sm text-warning-700 mt-2">{message}</p>
            <p className="text-xs text-warning-600 mt-2">
              本次未生成诊断结论，也不会计入诊断历史。
            </p>
          </div>
        </div>
      </div>
      <div className="flex flex-wrap gap-3">
        <button className="btn-primary" onClick={onReset}>
          <RotateCcw size={16} /> 重新上传图像
        </button>
      </div>
    </div>
  );
}

/* ---------- 报告步骤 ---------- */
function ReportStep({
  report,
  riskLevel,
  riskScope,
  recommendations,
  decisionReason,
  clarificationRounds,
  reviewRequired,
  conflictCount,
  sessionId,
  reportAt,
  metrics,
  specimenFile,
  specimenUrl,
  cells,
  chatOpen,
  onChat,
  onReset,
}: {
  report: string;
  riskLevel?: RiskLevel | null;
  riskScope?: 'cell' | 'case' | null;
  recommendations: string[];
  decisionReason?: string | null;
  clarificationRounds?: number;
  reviewRequired?: boolean;
  conflictCount?: number;
  sessionId?: string | null;
  reportAt?: number | null;
  chatOpen: boolean;
  metrics: {
    totalCells: number | null;
    abnormalCells: number | null;
    highestRisk: string | null;
    maxUncertainty: number | null;
    thresholds: { high: number; medium: number } | null;
    aggregation: string | null;
  };
  specimenFile?: File | null;
  specimenUrl?: string | null;
  cells?: CellResult[];
  onChat: () => void;
  onReset: () => void;
}) {
  const risk: RiskLevel = riskLevel ?? riskFromText(report);
  const [copied, setCopied] = useState(false);
  const doctor = getDoctor();

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(report);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* 剪贴板不可用（非安全上下文）时静默 */
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_260px] items-start">
      {/* ---------- 左：报告文档 ---------- */}
      <article className="card print-area">
        <div className="p-6 sm:p-8">
          {/* 文档抬头 */}
          <header className="pb-4 border-b border-hairline">
            <h2 className="title-serif text-[22px] leading-snug sm:text-[26px]">
              宫颈细胞学辅助判读报告
            </h2>
            <p className="mt-2 text-[12.5px] text-muted tabular">
              {sessionId ? `病例 ${sessionId.slice(0, 8)}` : '本机会话'}
              {reportAt ? ` · ${formatTime(reportAt)}` : ''}
              {' · 标本：液基细胞学'}
            </p>
          </header>

          {/* 口径提示 */}
          {riskScope === 'cell' && (
            <p className="mt-4 rounded-md border-l-2 border-risk-medium bg-risk-mediumBg px-3 py-2 text-[13px] leading-relaxed text-risk-mediumInk">
              本次仅检出少量细胞，风险等级属于<strong>细胞级提示</strong>，不代表病例整体风险。
            </p>
          )}

          {/* 结论冲突提示：必须出现在正文里，右侧栏打印时会被隐藏 */}
          {reviewRequired && (
            <p className="mt-4 rounded-md border-l-2 border-warning-400 bg-warning-50 px-3 py-2 text-[13px] leading-relaxed text-warning-800">
              本病例存在 <strong>{conflictCount ?? 0}</strong> 个结论冲突细胞
              （二分类判定为异常，而病理分级模型归为非异常类）。
              病例级结论仅供参考，<strong>请人工镜检复核后再出具诊断</strong>。
            </p>
          )}

          {/* 正文：后端返回 Markdown 标题时按章节渲染，否则整段呈现（不臆造章节） */}
          {(() => {
            const sections = parseReportSections(report);
            if (!sections) {
              return (
                <div className="mt-5 whitespace-pre-wrap text-[15px] leading-[1.8] text-ink">
                  {report}
                </div>
              );
            }
            return (
              <div className="mt-5 space-y-5">
                {sections.map((s, i) =>
                  s.title ? (
                    <section key={i}>
                      <h3 className="title-serif text-[15px] pt-4 border-t border-hairline first:border-0 first:pt-0">
                        {s.title}
                      </h3>
                      <div className="mt-2 whitespace-pre-wrap text-[15px] leading-[1.8] text-ink">
                        {s.body}
                      </div>
                    </section>
                  ) : (
                    <div
                      key={i}
                      className="whitespace-pre-wrap text-[15px] leading-[1.8] text-ink"
                    >
                      {s.body}
                    </div>
                  ),
                )}
              </div>
            );
          })()}

          {/* 建议 */}
          {recommendations.length > 0 && (
            <section className="mt-6 pt-5 border-t border-hairline">
              <h3 className="title-serif text-[15px]">处置建议</h3>
              <ul className="mt-2.5 list-disc pl-5 space-y-1.5 text-[14px] leading-relaxed text-ink">
                {recommendations.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            </section>
          )}

          {(specimenFile || specimenUrl) && cells && cells.length > 0 && (
            <section className="mt-6">
              <h3 className="title-serif text-[15px]">判读依据 · 视野与检测框</h3>
              <SpecimenView
                file={specimenFile ?? undefined}
                src={specimenUrl}
                cells={cells}
                alt="细胞学视野与检测框"
                className="mt-2 h-56 w-full rounded-md border border-hairline"
              />
              <p className="mt-1.5 text-[12px] text-muted">
                红框为异常细胞，绿框为正常细胞；框位与细胞表序号一致。
              </p>
            </section>
          )}

          {/* 局限性 + 请求者（不把 LLM 输出说成医生亲笔） */}
          <footer className="mt-6 pt-4 border-t border-hairline text-[12px] leading-relaxed text-muted">
            <p>
              本报告由 AI 系统基于图像分析自动生成，属于临床决策支持材料，不作为最终病理诊断。
              最终诊断须由执业医师结合临床与镜检复核确定。
            </p>
            <p className="mt-2">
              请求医生：
              <span className="text-ink">
                {doctor?.name ?? '未记录'}
                {doctor?.department ? ` · ${doctor.department}` : ''}
              </span>
              {sessionId && (
                <>
                  {' · '}会话 ID：
                  <span className="tabular">{sessionId.slice(0, 8)}</span>
                </>
              )}
            </p>
          </footer>
        </div>
      </article>

      {/* ---------- 右：病例上下文 ---------- */}
      <aside className="no-print space-y-4 lg:sticky lg:top-20">
        <div className="card p-4">
          <p className="text-[11.5px] tracking-[0.08em] text-muted">风险判定</p>
          <div className="mt-2 flex items-center gap-2 flex-wrap">
            <RiskBadge level={risk} size="md" />
            <RiskScopeTag scope={riskScope} level={risk} />
            <ReviewTag required={reviewRequired} count={conflictCount} />
          </div>

          {metrics.totalCells != null && (
            <dl className="mt-3 flex flex-wrap gap-x-4 gap-y-1 border-t border-hairline pt-3 text-[12px]">
              <div className="flex gap-1">
                <dt className="text-muted">细胞</dt>
                <dd className="tabular text-ink">{metrics.totalCells}</dd>
              </div>
              <div className="flex gap-1">
                <dt className="text-muted">异常</dt>
                <dd className="tabular text-ink">{metrics.abnormalCells ?? 0}</dd>
              </div>
              {metrics.maxUncertainty != null && (
                <div className="flex gap-1">
                  <dt className="text-muted">
                    不确定性 P{metrics.aggregation?.replace('p', '') ?? '90'}
                  </dt>
                  <dd className="tabular text-ink">{pct(metrics.maxUncertainty)}</dd>
                </div>
              )}
              {metrics.highestRisk && (
                <div className="flex gap-1">
                  <dt className="text-muted">最高类别</dt>
                  <dd className="text-ink">{classLabel(metrics.highestRisk)}</dd>
                </div>
              )}
            </dl>
          )}

          {decisionReason && (
            <div className="mt-3 pt-3 border-t border-hairline">
              <p className="text-[11.5px] text-muted">判读依据</p>
              <p className="mt-1 text-[12.5px] leading-relaxed text-ink">{decisionReason}</p>
            </div>
          )}

          {(clarificationRounds ?? 0) > 0 && (
            <p className="mt-2 text-[12px] text-muted">
              已完成 {clarificationRounds} 轮临床信息补充
            </p>
          )}
        </div>

        <div className="card p-4 space-y-2">
          <button className="btn-secondary w-full justify-start" onClick={() => window.print()}>
            <Printer size={15} /> 打印 / 存为 PDF
          </button>
          <button className="btn-secondary w-full justify-start" onClick={copy} disabled={!report}>
            <Copy size={15} /> {copied ? '已复制' : '复制报告全文'}
          </button>
          <button
            className={`w-full justify-start ${chatOpen ? 'btn-secondary' : 'btn-primary'}`}
            onClick={onChat}
            aria-expanded={chatOpen}
          >
            <MessageCircle size={15} /> {chatOpen ? '追问面板已打开' : '医生追问'}
          </button>
        </div>

        <button className="btn-ghost w-full justify-start text-[12.5px]" onClick={onReset}>
          <RotateCcw size={14} /> 重新诊断
        </button>
      </aside>
    </div>
  );
}

/* ---------- 追问步骤 ---------- */
function ChatStep({
  messages,
  sending,
  onSend,
}: {
  messages: ChatMessage[];
  sending: boolean;
  onSend: (input: string) => void;
}) {
  const [input, setInput] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  // 报告正文已在左侧文档中呈现，抽屉里不再重复一遍
  const thread = messages.filter((m) => m.type !== 'report');

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [thread.length, sending]);

  const send = () => {
    if (!input.trim() || sending) return;
    onSend(input.trim());
    setInput('');
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-3.5">
        {thread.length === 0 && (
          <p className="py-6 text-center text-[12.5px] leading-relaxed text-muted">
            可以询问判读依据、鉴别诊断，或下一步处置。
          </p>
        )}

        {thread.map((msg, i) => {
          const mine = msg.role === 'user';
          const system = msg.role === 'system';
          return (
            <div key={i} className={`flex ${mine ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[85%] rounded-md px-3 py-2 text-[13px] leading-relaxed ${
                  mine
                    ? 'bg-hematoxylin text-white'
                    : system
                    ? 'bg-glass text-muted'
                    : 'border border-hairline bg-white text-ink'
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.content}</div>
                <div
                  className={`mt-1 text-[10px] tabular ${
                    mine ? 'text-white/60' : 'text-muted/70'
                  }`}
                >
                  {formatTime(msg.time)}
                </div>
              </div>
            </div>
          );
        })}

        {sending && (
          <p className="animate-pulse-soft text-[12px] text-hematoxylin">AI 正在回答…</p>
        )}
        <div ref={endRef} />
      </div>

      <div className="mt-4 border-t border-hairline pt-3">
        <div className="flex items-end gap-2">
          <textarea
            className="input min-h-[44px] resize-none leading-relaxed"
            rows={2}
            placeholder={
              sending
                ? '请等待当前回答完成…'
                : '输入问题，Enter 发送，Shift+Enter 换行'
            }
            value={input}
            disabled={sending}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            aria-label="追问内容"
          />
          <button
            className="btn-primary h-11 shrink-0"
            onClick={send}
            disabled={!input.trim() || sending}
            aria-label="发送"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}