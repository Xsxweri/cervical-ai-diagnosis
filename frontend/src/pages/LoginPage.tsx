import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import axios from 'axios';
import { Eye, EyeOff } from 'lucide-react';
import { login } from '../libts/api';
import MicrographField from '../components/ui/MicrographField';

type ErrorKind = 'auth' | 'rate' | 'network' | 'expired' | 'field';

interface LoginError {
  kind: ErrorKind;
  text: string;
}

/** 与后端 core/auth.py 的 ACCESS_TOKEN_EXPIRE_HOURS 保持一致 */
const SESSION_HOURS = 1;
const VERSION = 'v2.1';

const FIELD_BASE =
  'h-12 w-full rounded-md border bg-white px-3.5 text-[16px] text-ink outline-none transition-colors ' +
  'placeholder:text-muted/60 hover:border-ink/30 ' +
  'focus:border-hematoxylin focus:ring-2 focus:ring-hematoxylin/30 ' +
  'disabled:bg-glass disabled:text-muted';

/** 载玻片抽象标记：圆角矩形 + 一角标签块 */
function BrandMark({ className = '' }: { className?: string }) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className={className}
    >
      <rect
        x="2.5"
        y="5.5"
        width="19"
        height="13"
        rx="2.5"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <path d="M14.5 5.5H19a2.5 2.5 0 0 1 2.5 2.5v2H14.5V5.5Z" fill="currentColor" opacity="0.85" />
    </svg>
  );
}

export default function LoginPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const usernameRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);
  const navTimer = useRef<number | null>(null);

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [entered, setEntered] = useState(false);
  const [error, setError] = useState<LoginError | null>(null);
  const [countdown, setCountdown] = useState(0);

  // 会话过期提示：由 http 拦截器跳转时带上 ?expired=1
  useEffect(() => {
    if (searchParams.get('expired')) {
      setError({ kind: 'expired', text: '登录已过期，请重新登录。' });
    }
  }, [searchParams]);

  // 429 倒计时
  useEffect(() => {
    if (countdown <= 0) return;
    const t = window.setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => window.clearTimeout(t);
  }, [countdown]);

  useEffect(
    () => () => {
      if (navTimer.current) window.clearTimeout(navTimer.current);
    },
    [],
  );

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading || countdown > 0) return;

    const name = username.trim();
    if (!name || !password) {
      setError({ kind: 'field', text: '请输入用户名和密码。' });
      (!name ? usernameRef : passwordRef).current?.focus();
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await login(name, password);
      setEntered(true);
      navTimer.current = window.setTimeout(() => navigate('/', { replace: true }), 250);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const status = err.response?.status;
        const detail = (err.response?.data as { detail?: string } | undefined)?.detail;

        if (!err.response) {
          setError({ kind: 'network', text: '无法连接判读服务。请检查网络或联系信息科。' });
        } else if (status === 401) {
          setError({ kind: 'auth', text: '用户名或密码不正确。' });
          passwordRef.current?.focus();
          passwordRef.current?.select();
        } else if (status === 429) {
          setError({ kind: 'rate', text: '尝试次数过多，请 1 分钟后再试。' });
          setCountdown(60);
        } else if (status === 403) {
          setError({ kind: 'auth', text: '账号无访问权限，请联系信息科。' });
        } else {
          setError({ kind: 'auth', text: detail ?? '登录失败，请稍后重试。' });
        }
      } else {
        setError({ kind: 'network', text: '无法连接判读服务。请检查网络或联系信息科。' });
      }
      setLoading(false);
    }
  };

  const busy = loading || countdown > 0;
  const describedBy = error ? 'login-error' : undefined;

  return (
    <div className="min-h-screen bg-glass text-ink md:flex">
      {/* ---------- 左：显微视野 ---------- */}
      <section className="relative h-[220px] shrink-0 overflow-hidden bg-ink md:h-auto md:w-[48%] xl:w-[56%]">
        <MicrographField />

        {/* 品牌字标 */}
        <div className="absolute inset-x-0 top-0 flex items-start gap-3 p-6 md:p-10">
          <BrandMark className="mt-0.5 shrink-0 text-white/90" />
          <div className="leading-tight">
            <p className="font-serif text-[17px] font-semibold tracking-[0.02em] text-white">
              宫颈细胞学
            </p>
            <p className="mt-1 text-[12px] tracking-[0.16em] text-white/[0.68]">辅助判读系统</p>
          </div>
        </div>

        {/* 标本说明：信息，不是装饰 */}
        <div className="absolute inset-x-0 bottom-0 flex items-end justify-between gap-4 p-6 md:p-10">
          <div>
            <p className="text-[12.5px] tracking-[0.03em] text-white/[0.78]">
              宫颈液基细胞学 · 40×
            </p>
            <p className="mt-1 text-[11px] text-white/[0.5]">示意视野（非真实病例）</p>
          </div>
          <div className="flex items-center gap-2 pb-1.5">
            <span className="h-px w-10 bg-white/60" aria-hidden="true" />
            <span className="font-mono text-[11px] tabular-nums text-white/[0.6]">50 µm</span>
          </div>
        </div>
      </section>

      {/* ---------- 右：载玻片标签式登录平面 ---------- */}
      <section className="flex flex-1 justify-center px-6 pb-14 pt-10 md:w-[52%] md:justify-start md:px-10 md:pt-[16vh] xl:w-[44%] xl:px-16">
        <div className="w-full max-w-[440px] md:max-w-[400px]">
          {/* 结构线：2px 苏木精蓝 + 右端伊红登记点 */}
          <div className="relative mb-8 h-[2px] w-full bg-hematoxylin" aria-hidden="true">
            <span className="absolute -top-[2px] right-0 h-[6px] w-[6px] rounded-full bg-eosin" />
          </div>

          <h1 className="font-serif text-[32px] font-semibold leading-[1.25] tracking-[0.01em] md:text-[40px] md:leading-[1.2]">
            回到判读台
          </h1>
          <p className="mt-3 text-[14px] leading-relaxed text-muted">
            登录后进入宫颈细胞学辅助判读工作台。
          </p>

          <form onSubmit={submit} className="mt-8 space-y-5" noValidate>
            {/* 用户名 */}
            <div>
              <label
                htmlFor="login-username"
                className="mb-2 block text-[13px] font-medium text-ink"
              >
                用户名
              </label>
              <input
                id="login-username"
                ref={usernameRef}
                className={`${FIELD_BASE} ${error ? 'border-eosin' : 'border-hairline'}`}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                autoCapitalize="none"
                spellCheck={false}
                disabled={busy}
                aria-invalid={!!error}
                aria-describedby={describedBy}
              />
            </div>

            {/* 密码 */}
            <div>
              <label
                htmlFor="login-password"
                className="mb-2 block text-[13px] font-medium text-ink"
              >
                密码
              </label>
              <div className="relative">
                <input
                  id="login-password"
                  ref={passwordRef}
                  type={showPassword ? 'text' : 'password'}
                  className={`${FIELD_BASE} pr-12 ${error ? 'border-eosin' : 'border-hairline'}`}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  disabled={busy}
                  aria-invalid={!!error}
                  aria-describedby={describedBy}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  disabled={busy}
                  aria-label={showPassword ? '隐藏密码' : '显示密码'}
                  aria-pressed={showPassword}
                  className="absolute right-[2px] top-[2px] flex h-11 w-11 items-center justify-center rounded text-muted transition-colors hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-hematoxylin/40 disabled:opacity-40"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {/* 错误区：伊红左线 + 文字（不抖动） */}
            {error && (
              <div
                id="login-error"
                role="alert"
                className="animate-fade-in border-l-2 border-eosin pl-3 text-[13px] leading-relaxed text-eosin"
              >
                {error.text}
                {error.kind === 'rate' && countdown > 0 && (
                  <span className="ml-1 font-mono tabular-nums">（{countdown}s）</span>
                )}
              </div>
            )}

            {/* 主按钮：宽度不变，加载中跑不确定进度线 */}
            <button
              type="submit"
              disabled={busy}
              className="relative h-12 w-full overflow-hidden rounded-md bg-hematoxylin text-[15px] font-medium text-white transition-colors hover:bg-hematoxylin/[0.92] focus:outline-none focus-visible:ring-2 focus-visible:ring-hematoxylin/40 focus-visible:ring-offset-2 focus-visible:ring-offset-glass disabled:cursor-not-allowed disabled:opacity-60"
            >
              <span>
                {entered ? '已进入' : loading ? '正在验证身份…' : '登录工作台'}
              </span>
              {loading && !entered && (
                <span className="absolute inset-x-0 bottom-0 h-[2px] overflow-hidden">
                  <span className="animate-progress-line block h-full w-1/3 bg-white/80" />
                </span>
              )}
            </button>
          </form>

          {/* 信任区 */}
          <div className="mt-8 border-t border-hairline pt-5">
            <p className="text-[12.5px] leading-relaxed text-muted">
              仅限授权医疗人员使用。登录与操作均记入审计日志。
            </p>
            <div className="mt-3 flex items-center justify-between gap-4">
              <p className="text-[12.5px] text-hematoxylin">忘记密码？联系信息科</p>
              <span className="font-mono text-[11.5px] tabular-nums text-muted/70">{VERSION}</span>
            </div>
            <p className="mt-1.5 text-[12.5px] text-muted/80">
              登录有效期 {SESSION_HOURS} 小时
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}