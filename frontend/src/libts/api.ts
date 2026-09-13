import axios from 'axios';
import type { AgentResponse, CVResult, FollowUpResponse, HealthResponse } from '../types';
import {getToken, setToken, clearToken} from './storage';

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
      if (error.response?.status === 401) clearToken();
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
    return data;
}

/** 健康检查 */
export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await http.get<HealthResponse>('/health');
  return data;
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
export async function agentUpload(file: File): Promise<AgentResponse> {
  const fd = new FormData();
  fd.append('file', file);
  const { data } = await http.post<AgentResponse>('/api/agent/upload', fd);
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
  const url = `${WS_BASE}/ws/agent/chat/${sessionId}?token=${encodeURIComponent(token ?? '')}`;
  const ws = new WebSocket(url);
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
  // 握手被后端以 4401 关闭时给出明确提示（token 缺失/过期）
  ws.onclose = (event) => {
    if (event.code === 4401) {
      clearToken();
      opts.onError?.('WebSocket 鉴权失败，请重新登录');
    }
  };
  return ws;
}