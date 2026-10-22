import { useEffect, useRef, useState } from 'react';
import { Send, AlertTriangle, Sparkles, RotateCcw, MessageSquareText, MessageCircle, X } from 'lucide-react';
import { useDiagnosis, type ChatMessage } from '../hooks/useDiagnosis';
import { riskFromText, formatTime } from '../libts/format';
import type { RiskLevel } from '../types';
import ImageDropzone from '../components/ui/ImageDropzone';
import Spinner from '../components/ui/Spinner';
import RiskBadge from '../components/ui/RiskBadge';
import RiskScopeTag from '../components/ui/RiskScopeTag';

const STEPS = ['上传图像', 'AI 分析', '报告', '追问'];

export default function DiagnosePage() {
  const diag = useDiagnosis();
  const [file, setFile] = useState<File | null>(null);
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
      {diag.step === 'analyzing' && <AnalyzingStep onCancel={diag.cancel} />}
      {diag.step === 'clarification' && (
        <ClarificationStep questions={diag.questions} loading={diag.isLoading} onSubmit={diag.submitClarification} />
      )}
      {diag.step === 'unreadable' && (
        <UnreadableStep
          message={diag.errorMessage ?? '未检测到可判读的宫颈细胞。'}
          onReset={diag.reset}
        />
      )}
      {diag.step === 'report' && (
        <ReportStep
          report={diag.report}
          riskLevel={diag.riskLevel}
          riskScope={diag.riskScope}
          recommendations={diag.recommendations}
          onChat={diag.startChat}
          onReset={diag.reset}
        />
      )}
      {diag.step === 'chat' && (
        <ChatStep
          messages={diag.messages}
          sending={diag.isSending}
          onSend={diag.sendMessage}
          onReset={diag.reset}
        />
      )}
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
const PIPELINE = [
  'YOLOv8 细胞检测',
  '二分类筛查（正常 / 异常）',
  '多分类细粒度分型',
  'MC Dropout 不确定性评估',
  'Agent 生成诊断报告',
];

function AnalyzingStep({ onCancel }: { onCancel: () => void }) {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => {
      setIdx((i) => {
        if (i >= PIPELINE.length - 1) {
          clearInterval(timer);
          return i;
        }
        return i + 1;
      });
    }, 1100);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="max-w-xl mx-auto card p-8 animate-fade-in">
      <div className="flex flex-col items-center gap-5">
        <Spinner size={44} />
        <h3 className="font-semibold text-gray-900">AI 正在分析细胞图像...</h3>
        <div className="w-full space-y-2.5 mt-2">
          {PIPELINE.map((label, i) => (
            <div key={label} className="flex items-center gap-3 text-sm">
              <div
                className={`w-4 h-4 rounded-full border-2 shrink-0 transition-colors ${
                  i < idx
                    ? 'bg-success-500 border-success-500'
                    : i === idx
                    ? 'border-primary-500 animate-pulse-soft'
                    : 'border-gray-200'
                }`}
              />
              <span className={i <= idx ? 'text-gray-700' : 'text-gray-400'}>{label}</span>
              {i === idx && <span className="ml-auto text-xs text-primary-500 animate-pulse-soft">进行中</span>}
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-400">
          进度为阶段提示，实际耗时取决于图像复杂度与服务器负载
        </p>
        <button className="btn-secondary" onClick={onCancel}>
          <X size={15} /> 取消分析
        </button>
      </div>
    </div>
  );
}

/* ---------- 澄清步骤 ---------- */
function ClarificationStep({
  questions,
  loading,
  onSubmit,
}: {
  questions: string[];
  loading: boolean;
  onSubmit: (answers: string[]) => void;
}) {
  const [answers, setAnswers] = useState<string[]>(() => new Array(questions.length).fill(''));
  const valid = answers.every((a) => a.trim());

  return (
    <div className="max-w-2xl mx-auto space-y-5 animate-fade-in">
      <div className="card p-5 border-warning-200 bg-warning-50/60">
        <div className="flex items-start gap-3">
          <AlertTriangle size={20} className="text-warning-600 mt-0.5 shrink-0" />
          <div>
            <h3 className="font-semibold text-warning-800">诊断不确定性较高</h3>
            <p className="text-sm text-warning-700 mt-1">
              AI 检测到部分细胞特征不明确，需要补充临床信息以提高诊断准确性。
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {questions.map((q, i) => (
          <div key={i} className="card p-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {i + 1}. {q}
            </label>
            <input
              className="input"
              placeholder="请输入..."
              value={answers[i] ?? ''}
              onChange={(e) => {
                const next = [...answers];
                next[i] = e.target.value;
                setAnswers(next);
              }}
            />
          </div>
        ))}
      </div>

      <button
        className="btn-primary w-full"
        disabled={!valid || loading}
        onClick={() => onSubmit(answers)}
      >
        {loading ? (
          <Spinner size={16} />
        ) : (
          <>
            <Sparkles size={16} /> 提交并重新诊断
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
  onChat,
  onReset,
}: {
  report: string;
  riskLevel?: RiskLevel | null;
  riskScope?: 'cell' | 'case' | null;
  recommendations: string[];
  onChat: () => void;
  onReset: () => void;
}) {
  const risk: RiskLevel = riskLevel ?? riskFromText(report);
  return (
    <div className="max-w-3xl mx-auto space-y-5 animate-fade-in">
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
          <h3 className="font-semibold text-gray-900 text-lg">📝 AI 诊断报告</h3>
          <div className="flex items-center gap-2">
            <RiskScopeTag scope={riskScope} level={risk} />
            <RiskBadge level={risk} size="md" />
          </div>
        </div>
        {riskScope === 'cell' && (
          <p className="mb-3 text-sm text-warning-800 bg-warning-50 rounded-lg p-3">
            本次仅检出少量细胞，风险等级属于<strong>细胞级提示</strong>，不代表病例整体风险。
          </p>
        )}
        <div className="whitespace-pre-wrap text-gray-700 leading-relaxed text-[15px] bg-gray-50 rounded-lg p-5">
          {report}
        </div>
        {recommendations.length > 0 && (
          <div className="mt-4">
            <p className="text-xs text-gray-400 mb-2">建议</p>
            <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700">
              {recommendations.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-3">
        <button className="btn-primary" onClick={onChat}>
          <MessageCircle size={16} /> 进入医生追问模式
        </button>
        <button className="btn-secondary" onClick={onReset}>
          <RotateCcw size={16} /> 重新诊断
        </button>
      </div>
    </div>
  );
}

/* ---------- 追问步骤 ---------- */
function ChatStep({
  messages,
  sending,
  onSend,
  onReset,
}: {
  messages: ChatMessage[];
  sending: boolean;
  onSend: (input: string) => void;
  onReset: () => void;
}) {
  const [input, setInput] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = () => {
    if (!input.trim() || sending) return;
    onSend(input.trim());
    setInput('');
  };

  return (
    <div className="max-w-3xl mx-auto card flex flex-col h-[600px] animate-fade-in">
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100">
        <div className="flex items-center gap-2 text-gray-700">
          <MessageSquareText size={18} className="text-primary-600" />
          <span className="font-medium">医生追问模式</span>
        </div>
        <button className="btn-ghost text-xs" onClick={onReset}>
          <RotateCcw size={14} /> 结束会话
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-primary-600 text-white'
                  : msg.role === 'system'
                  ? 'bg-gray-100 text-gray-500 text-xs'
                  : msg.type === 'report'
                  ? 'bg-green-50 border border-green-200 text-gray-800'
                  : 'bg-white border border-gray-200 text-gray-800'
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>
              <div className={`mt-1 text-[10px] ${msg.role === 'user' ? 'text-primary-200' : 'text-gray-400'}`}>
                {formatTime(msg.time)}
              </div>
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="border-t border-gray-100 p-4 bg-gray-50/60 rounded-b-xl">
        {sending && (
          <p className="mb-2 text-xs text-primary-600 animate-pulse-soft">AI 正在回答...</p>
        )}
        <div className="flex gap-2">
          <input
            className="input"
            placeholder={sending ? '请等待当前回答完成...' : '输入追问内容，Enter 发送...'}
            value={input}
            disabled={sending}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button
            className="btn-primary shrink-0"
            onClick={send}
            disabled={!input.trim() || sending}
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}