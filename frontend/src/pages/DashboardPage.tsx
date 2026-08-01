import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  ScanSearch,
  History,
  Server,
  Cpu,
  Boxes,
  Gauge,
  ArrowRight,
} from 'lucide-react';
import { fetchHealth } from '../libts/api';
import { getHistory } from '../libts/storage';
import { formatTime } from '../libts/format';
import type { HealthResponse } from '../types';
import StatCard from '../components/ui/StatCard';
import RiskBadge from '../components/ui/RiskBadge';
import Spinner from '../components/ui/Spinner';
import EmptyState from '../components/ui/EmptyState';

const QUICK_LINKS = [
  { to: '/diagnose', label: '智能诊断', desc: 'AI-Agent 全流程辅助诊断', icon: Sparkles, tone: 'ai' as const },
  { to: '/analyze', label: '快速分析', desc: '单张 / 批量 / ZIP 细胞分析', icon: ScanSearch, tone: 'primary' as const },
  { to: '/history', label: '诊断历史', desc: '查看本地历史记录', icon: History, tone: 'gray' as const },
];

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);
  const recent = getHistory().slice(0, 5);

  useEffect(() => {
    let cancelled = false;
    fetchHealth()
      .then((h) => {
        if (!cancelled) setHealth(h);
      })
      .catch(() => {
        if (!cancelled) setHealthError('无法连接后端服务（localhost:8000）');
      })
      .finally(() => {
        if (!cancelled) setHealthLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-6">
      {/* 欢迎横幅 */}
      <div className="rounded-2xl bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 text-white p-6 sm:p-8 animate-fade-in">
        <h2 className="text-xl sm:text-2xl font-bold">欢迎使用 AI-Agent 宫颈病理诊断系统</h2>
        <p className="mt-2 text-primary-100 text-sm sm:text-base max-w-2xl">
          基于 YOLOv8 细胞检测、ResNet50 双级分类与 LangGraph 智能体工作流，
          提供从图像上传、自动诊断到医生追问的完整闭环。
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <Link to="/diagnose" className="btn bg-white text-primary-700 hover:bg-primary-50">
            <Sparkles size={16} /> 开始智能诊断
          </Link>
          <Link to="/analyze" className="btn bg-primary-800/40 text-white hover:bg-primary-800/60 border border-white/20">
            <ScanSearch size={16} /> 快速分析
          </Link>
        </div>
      </div>

      {/* 快捷入口 */}
      <div className="grid sm:grid-cols-3 gap-4">
        {QUICK_LINKS.map(({ to, label, desc, icon, tone }) => (
          <Link key={to} to={to} className="card card-hover p-5 group">
            <div className="flex items-start justify-between">
              <StatCardStandalone icon={icon} label={label} desc={desc} tone={tone} />
              <ArrowRight size={18} className="text-gray-300 group-hover:text-primary-500 transition-colors" />
            </div>
          </Link>
        ))}
      </div>

      {/* 系统状态 + 最近诊断 */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="card p-6 lg:col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <Server size={18} className="text-primary-600" />
            <h3 className="font-semibold text-gray-900">系统状态</h3>
          </div>

          {healthLoading && <Spinner label="正在获取系统状态..." />}

          {healthError && !healthLoading && (
            <EmptyState
              icon={Server}
              title="后端服务未连接"
              description={healthError}
              action={<Link to="/diagnose" className="btn-primary">前往诊断页重试</Link>}
            />
          )}

          {health && !healthLoading && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <StatCard icon={Cpu} label="运行设备" value={health.device} tone="primary" />
                <StatCard icon={Gauge} label="MC Dropout 采样" value={health.config.mc_dropout_samples} sub="次" tone="ai" />
                <StatCard icon={Boxes} label="模型状态" value="已加载" tone="success" />
              </div>
              <div className="rounded-lg bg-gray-50 p-4 text-xs space-y-1.5 text-gray-600">
                <p><span className="text-gray-400">检测模型：</span>{health.models.yolo}</p>
                <p><span className="text-gray-400">二分类：</span>{health.models.二分类}</p>
                <p><span className="text-gray-400">多分类：</span>{health.models.多分类}</p>
                <p>
                  <span className="text-gray-400">阈值：</span>
                  不确定 {health.config.不确定阈值} · 异常 CONF {health.config.异常的CONF阈值}
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <History size={18} className="text-primary-600" />
            <h3 className="font-semibold text-gray-900">最近诊断</h3>
          </div>
          {recent.length === 0 ? (
            <EmptyState icon={History} title="暂无诊断记录" description="完成一次智能诊断或快速分析后，记录会出现在这里" />
          ) : (
            <ul className="space-y-3">
              {recent.map((r) => (
                <li key={r.id} className="flex items-center justify-between gap-2 text-sm">
                  <div className="min-w-0">
                    <p className="truncate text-gray-800 font-medium">{r.filename}</p>
                    <p className="text-xs text-gray-400">
                      {formatTime(r.timestamp)} · {r.mode === 'agent' ? '智能诊断' : '快速分析'}
                    </p>
                  </div>
                  <RiskBadge level={r.riskLevel} />
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

/* 内联小卡片（避免过度抽象） */
function StatCardStandalone({ icon: Icon, label, desc, tone }: { icon: any; label: string; desc: string; tone: string }) {
  const map: Record<string, string> = {
    ai: 'bg-ai-50 text-ai-600',
    primary: 'bg-primary-50 text-primary-600',
    gray: 'bg-gray-100 text-gray-500',
  };
  return (
    <div className="flex items-start gap-3">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${map[tone] ?? map.gray}`}>
        <Icon size={20} />
      </div>
      <div>
        <p className="font-semibold text-gray-900">{label}</p>
        <p className="text-xs text-gray-400 mt-0.5">{desc}</p>
      </div>
    </div>
  );
}