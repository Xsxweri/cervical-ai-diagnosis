import axios from 'axios';
import type { AgentResponse, CVResult, FollowUpResponse, HealthResponse, SessionResponse, SystemInfoResponse, } from '../types';
import { getToken, setToken, clearToken, setDoctor } from './storage';

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000';
export const WS_BASE: string =
  (import.meta.env.VITE_WS_BASE as string | undefined) ?? 'ws://localhost:8000';

const http = axios.create({ baseURL: API_BASE, timeout: 120_000 });

http.interceptors.request.use((config) => {
    const token = getToken();
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });

http.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      clearToken();
      // 登录请求本身的 401 不跳转（路径已在 /login）；其余视为会话过期
      if (!window.location.pathname.startsWith('/login')) {
        window.location.replace('/login?expired=1');
      }
    }
    return Promise.reject(error);
  }
);

export interface LoginResponse {
  token: string;
  doctor: {name?:string, hospital?:string, department?:string};
}

/**医生登录并保存token */
export async function login(username: string, password: string): Promise<LoginResponse> {
  const { data } = await http.post<LoginResponse>('/api/auth/login', { username, password });
  setToken(data.token);
  setDoctor(data.doctor ?? null);
  return data;
}

/** 存活探针（无鉴权） */
export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await http.get<HealthResponse>('/health');
  return data;
}

/** 系统信息（需鉴权）：设备、模型路径、阈值、RAG 可用性 */
export async function fetchSystemInfo(): Promise<SystemInfoResponse> {
  const { data } = await http.get<SystemInfoResponse>('/api/system/info');
  return data;
}

/** 读取会话的完整结果（Redis TTL 内有效，受会话归属校验） */
export async function fetchSession(sessionId: string): Promise<SessionResponse> {
  const { data } = await http.get<SessionResponse>(
    `/api/agent/session/${encodeURIComponent(sessionId)}`,
  );
  return data;
}

/**
 * 读取会话原图，返回 object URL。
 *
 * 用 blob 而非直接给 <img src>：鉴权走 Authorization 头，
 * 若拼成 URL 传 token，会让凭证进入浏览器历史与服务端访问日志。
 */
export async function fetchSessionImage(sessionId: string): Promise<string> {
  const { data } = await http.get<Blob>(
    `/api/agent/session/${encodeURIComponent(sessionId)}/image`,
    { responseType: 'blob' },
  );
  return URL.createObjectURL(data);
}

/** 单张图片快速分析 */
export async function analyzeImage(file: File): Promise<CVResult> {
  const fd = new FormData();
  fd.append('file', file);
  const { data } = await http.post<CVResult>('/analyze', fd);
  return data;
}

/** 多张图片批量分析 */
export async function analyzeBatch(files: File[]): Promise<Record<string, CVResult>> {
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  const { data } = await http.post<Record<string, CVResult>>('/analyze_batch', fd);
  return data;
}

/** ZIP 压缩包批量分析 */
export async function analyzeZip(file: File): Promise<Record<string, CVResult>> {
  const fd = new FormData();
  fd.append('upload_file', file);
  const { data } = await http.post<Record<string, CVResult>>('/analyze_zip', fd);
  return data;
}

/** Agent：上传图像并启动诊断 */
export async function agentUpload(file: File, signal?: AbortSignal): Promise<AgentResponse> {
  const fd = new FormData();
  fd.append('file', file);
  const { data } = await http.post<AgentResponse>('/api/agent/upload', fd, { signal });
  return data;
}

/** Agent：提交澄清回答 */
export async function agentClarify(sessionId: string, userInput: string): Promise<AgentResponse> {
  const { data } = await http.post<AgentResponse>(`/api/agent/clarify/${sessionId}`, {
    user_input: userInput,
  });
  return data;
}

/** Agent：HTTP 追问（WebSocket 不可用时降级） */
export async function agentFollowUp(sessionId: string, message: string): Promise<string> {
  const { data } = await http.post<FollowUpResponse>('/api/agent/followup', {
    session_id: sessionId,
    message,
  });
  return data.response;
}

export interface WSOptions {
  onStream: (content: string) => void;
  onComplete?: () => void;
  onError?: (message: string) => void;
}

/** 建立 Agent WebSocket 会话 */
export function connectAgentWS(sessionId: string, opts: WSOptions): WebSocket {
  const token = getToken();
  /**
   * 关于鉴权方式的说明：
   * 浏览器提供 Sec-WebSocket-Protocol 子协议时，服务端必须回选其中一个，
   * 否则按规范握手失败（Chrome: "Sent non-empty 'Sec-WebSocket-Protocol'
   * header but no response was received"，随后以 1006 关闭）。
   * 后端握手未回传子协议，因此这里不再发送子协议，改用后端同样支持的
   * token 查询参数；若后端改为 accept(subprotocol="bearer")，可切回
   * `new WebSocket(url, ['bearer', token])`。
   */
  const url = `${WS_BASE}/ws/agent/chat/${encodeURIComponent(sessionId)}`;
  const ws = new WebSocket(url, token ? ['bearer', token] : ['bearer']);
  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data) as { type: string; content?: string; message?: string };
      if (msg.type === 'stream' && msg.content) opts.onStream(msg.content);
      else if (msg.type === 'complete') opts.onComplete?.();
      else if (msg.type === 'error') opts.onError?.(msg.message ?? '未知错误');
    } catch {
      /* 忽略非 JSON 消息 */
    }
  };
  ws.onclose = (event) => {
    if (event.code === 4401) {
      clearToken();
      opts.onError?.('WebSocket 鉴权失败，请重新登录');
    } else if (event.code === 1006) {
      // 握手/连接异常：不静默，否则用户会误以为流式输出"就是不出现"
      opts.onError?.('实时通道连接失败，已切换为普通追问模式（无流式输出）');
    }
  };
  return ws;
}