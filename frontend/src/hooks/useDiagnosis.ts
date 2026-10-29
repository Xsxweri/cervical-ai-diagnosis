import { useCallback, useEffect, useRef, useState } from 'react';
import type { AgentResponse, CellResult, RiskLevel, SessionResponse, TaskAccepted } from '../types';
import { agentClarify, agentFollowUp, agentUpload, connectAgentWS, fetchSession, fetchSessionImage, fetchSessionGradcam, fetchTaskStatus, generateSessionGradcam } from '../libts/api';
import { riskFromText } from '../libts/format';
import { addHistory } from '../libts/storage';

export type DiagnosisStep =
  | 'upload'
  | 'analyzing'
  | 'clarification'
  | 'report'
  | 'unreadable'
  | 'chat';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  type?: 'text' | 'report';
  time: number;
}

export function useDiagnosis() {
  const [step, setStep] = useState<DiagnosisStep>('upload');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [questions, setQuestions] = useState<string[]>([]);
  const [report, setReport] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [riskLevel, setRiskLevel] = useState<RiskLevel | null>(null);
  const [riskScope, setRiskScope] = useState<'cell' | 'case' | null>(null);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [decisionReason, setDecisionReason] = useState<string | null>(null);
  const [clarificationRounds, setClarificationRounds] = useState(0);
  const [reviewRequired, setReviewRequired] = useState(false);
  const [conflictCount, setConflictCount] = useState(0);
  const [reportAt, setReportAt] = useState<number | null>(null);
  const [specimenUrl, setSpecimenUrl] = useState<string | null>(null);
  const [gradcamUrl, setGradcamUrl] = useState<string | null>(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [gradcamError, setGradcamError] = useState<string | null>(null);
  const [cvCells, setCvCells] = useState<CellResult[]>([]);
  const [cvMetrics, setCvMetrics] = useState<{
    totalCells: number | null;
    abnormalCells: number | null;
    highestRisk: string | null;
    maxUncertainty: number | null;
    thresholds: { high: number; medium: number } | null;
    aggregation: string | null;
  }>({
    totalCells: null,
    abnormalCells: null,
    highestRisk: null,
    maxUncertainty: null,
    thresholds: null,
    aggregation: null,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const filenameRef = useRef('病例');

  const pushMessage = useCallback((msg: Omit<ChatMessage, 'time'>) => {
    setMessages((prev) => [...prev, { ...msg, time: Date.now() }]);
  }, []);

  const connectWS = useCallback(
    (sid: string) => {
      wsRef.current?.close();
      wsRef.current = connectAgentWS(sid, {
        onStream: (content) => {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last?.role === 'assistant' && last.type === 'text') {
              return [...prev.slice(0, -1), { ...last, content: last.content + content }];
            }
            return [...prev, { role: 'assistant', content, type: 'text', time: Date.now() }];
          });
        },
        onError: (message) => pushMessage({ role: 'system', content: `错误: ${message}` }),
      });
    },
    [pushMessage]
  );

    const finishReport = useCallback(
    (
      res: AgentResponse,
      filename: string,
      opts: { record?: boolean; connect?: boolean } = {},
    ) => {
      const { record = true, connect = true } = opts;
      const text = res.report ?? '';

      setReport(text);
      setRecommendations(res.recommendations ?? []);
      setRiskScope(res.risk_scope ?? null);
      setDecisionReason(res.decision_reason ?? null);
      setClarificationRounds(res.clarification_rounds ?? 0);
      setReviewRequired(res.review_required ?? false);
      setConflictCount(res.conflict_count ?? res.grading_conflict_count ?? 0);
      setCvMetrics({
        totalCells: res.total_cells ?? null,
        abnormalCells: res.abnormal_cells ?? null,
        highestRisk: res.highest_risk ?? null,
        maxUncertainty: res.max_uncertainty ?? null,
        thresholds: res.uncertainty_thresholds ?? null,
        aggregation: res.uncertainty_aggregation ?? null,
      });
      setCvCells(res.cells ?? []);

      // 无法判读（0 细胞等）：单独一步展示原因，绝不落到"低风险"，也不写历史
      if (res.status === 'unreadable' || res.error_message) {
        const reason = res.error_message ?? '未检测到可判读的宫颈细胞，无法生成诊断结论。';
        setErrorMessage(reason);
        setRiskLevel('unreadable');
        setStep('unreadable');
        pushMessage({ role: 'system', content: `无法判读：${reason}` });
        return;
      }

      const backRisk =
        res.risk_level && res.risk_level !== 'pending' ? res.risk_level : null;
      const finalRisk = backRisk ?? riskFromText(text);
      setRiskLevel(finalRisk);
      setReportAt(Date.now());
      setStep('report');
      pushMessage({ role: 'assistant', content: text, type: 'report' });

      if (record) {
        addHistory({
          id: res.session_id,
          timestamp: Date.now(),
          filename,
          mode: 'agent',
          riskLevel: finalRisk,
          riskScope: res.risk_scope ?? null,
          reviewRequired: res.review_required ?? false,
          sessionId: res.session_id,
          reportPreview: text.slice(0, 200),
        });
      }
      if (connect) connectWS(res.session_id);
    },
    [connectWS, pushMessage],
  );

  const POLL_MAX_MS = 10 * 60 * 1000;

  function isAccepted(r: unknown): r is TaskAccepted {
  return !!r && typeof r === 'object' && (r as TaskAccepted).task_status === 'queued'
      && !('cells' in (r as object));
}
  /** 轮询直到 done / failed / 超时 / 用户取消 */
  async function waitForTask(
    sessionId: string,
    signal: AbortSignal,
  ): Promise<SessionResponse> {
    const started = Date.now();
    await new Promise((r) => setTimeout(r, 1500));
    for (;;) {
      if (signal.aborted) throw new DOMException('aborted', 'AbortError');
      if (Date.now() - started > POLL_MAX_MS) {
        throw new Error('诊断超时（超过 10 分钟），请稍后在历史记录中查看');
      }
      const st = await fetchTaskStatus(sessionId);
      if (st.task_status === 'done') return st;
      if (st.task_status === 'failed') throw new Error(st.error || '诊断失败');
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  const start = useCallback(
    async (file: File) => {
      filenameRef.current = file.name;
      setStep('analyzing');
      setIsLoading(true);
      setErrorMessage(null);
      setRiskScope(null);
      setRecommendations([]);
      setDecisionReason(null);
      setClarificationRounds(0);
      setReviewRequired(false);
      setConflictCount(0);
      pushMessage({ role: 'system', content: `已上传图像：${file.name}` });
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const raw = await agentUpload(file, controller.signal);
        const res = isAccepted(raw)
          ? await waitForTask(raw.session_id, controller.signal)
          : (raw as AgentResponse);
        setSessionId(res.session_id);
        if (res.status === 'awaiting_clarification' && res.questions?.length) {
          setQuestions(res.questions);
          setStep('clarification');
          pushMessage({ role: 'system', content: 'AI 检测到诊断不确定性较高，需要补充信息。' });
          res.questions.forEach((q, i) =>
            pushMessage({ role: 'assistant', content: `${i + 1}. ${q}` })
          );
        } else {
          finishReport(res, file.name);
        }
      } catch (e) {
        // 用户主动取消：回到上传页，不当作失败
        if (controller.signal.aborted) {
          setStep('upload');
          return;
        }
        pushMessage({
          role: 'system',
          content: `诊断失败：${e instanceof Error ? e.message : '未知错误'}`,
        });
        setStep('upload');
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
        setIsLoading(false);
      }
    },
    [finishReport, pushMessage]
  );

  /** 从服务端会话恢复一次完整诊断（用于历史记录"打开报告 **/
  const loadSession = useCallback(
    async (sid: string) => {
      setStep('analyzing');
      setIsLoading(true);
      setErrorMessage(null);
      setSpecimenUrl(null);
      try {
        const res = await fetchSession(sid);
        setSessionId(res.session_id);
        // 恢复时不重复写历史、不自动连 WS（用户可能只想看报告）
        finishReport(res, '已保存会话', { record: false, connect: false });

        if (res.image_available) {
          try {
            const url = await fetchSessionImage(sid);
            setSpecimenUrl(url);
          } catch {
            // 原图可能已被清理：报告仍可阅读，只是缺少证据图
            setSpecimenUrl(null);
          }
        }
        try {
          setGradcamUrl(await fetchSessionGradcam(sid));
        } catch {
          setGradcamUrl(null);
        }
      } catch (e) {
        const status = (e as { response?: { status?: number } }).response?.status;
        const reason =
          status === 404
            ? '该会话已过期（服务端仅保留 1 天）或已被清理，无法恢复完整报告。'
            : status === 403
            ? '无权访问该会话。'
            : `读取会话失败：${e instanceof Error ? e.message : '未知错误'}`;
        setErrorMessage(reason);
        setStep('unreadable');
        pushMessage({ role: 'system', content: reason });
      } finally {
        setIsLoading(false);
      }
    },
    [finishReport, pushMessage],
  );

  /** 取消正在进行的分析 */
  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsLoading(false);
    setStep('upload');
  }, []);

  const submitClarification = useCallback(
    async (answers: string[]) => {
      if (!sessionId) return;
      setIsLoading(true);
      const answerText = answers.join('；');
      pushMessage({ role: 'user', content: answerText });
      try {
        const res = await agentClarify(sessionId, answerText);
        finishReport(res, filenameRef.current);
      } catch (e) {
        pushMessage({
          role: 'system',
          content: `澄清处理失败：${e instanceof Error ? e.message : '未知错误'}`,
        });
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, finishReport, pushMessage]
  );

  /** 报告页 → 进入医生追问模式（补连 WS 若未连接） */
  const startChat = useCallback(() => {
    if (!sessionId) return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connectWS(sessionId);
    }
    setStep('chat');
  }, [sessionId, connectWS]);

  /** 关闭追问抽屉，回到报告视图（报告内容不变） */
  const closeChat = useCallback(() => {
    setStep((s) => (s === 'chat' ? 'report' : s));
  }, []);

  const sendMessage = useCallback(
    async (input: string) => {
      if (!sessionId || isSending) return;
      pushMessage({ role: 'user', content: input });
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ message: input }));
        return;
      }
      // HTTP 降级：必须有"正在回答"状态，否则用户会重复发送
      setIsSending(true);
      try {
        const resp = await agentFollowUp(sessionId, input);
        pushMessage({ role: 'assistant', content: resp, type: 'text' });
      } catch (e) {
        pushMessage({
          role: 'system',
          content: `追问失败：${e instanceof Error ? e.message : '未知错误'}`,
        });
      } finally {
        setIsSending(false);
      }
    },
    [isSending, sessionId, pushMessage]
  );

  /** 按需生成并加载 Grad-CAM（用户点击时触发） */
  const requestGradcam = useCallback(async () => {
    if (!sessionId || gradcamLoading || gradcamUrl) return;
    setGradcamLoading(true);
    setGradcamError(null);
    try {
      await generateSessionGradcam(sessionId);
      setGradcamUrl(await fetchSessionGradcam(sessionId));
    } catch (e) {
      // 解析后端 detail（axios 错误体）
      const detail =
        (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setGradcamError(detail ?? '热力图生成失败，请稍后重试');
    } finally {
      setGradcamLoading(false);
    }
  }, [sessionId, gradcamLoading, gradcamUrl]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    setStep('upload');
    setRiskLevel(null);
    setRiskScope(null);
    setRecommendations([]);
    setErrorMessage(null);
    setIsSending(false);
    setMessages([]);
    setSessionId(null);
    setQuestions([]);
    setReport('');
    setDecisionReason(null);
    setClarificationRounds(0);
    setReviewRequired(false);
    setConflictCount(0);
    setReportAt(null);
    setSpecimenUrl(null);
    setGradcamUrl(null);
    setGradcamLoading(false);
    setGradcamError(null);
    setCvCells([]);
    setCvMetrics({
      totalCells: null,
      abnormalCells: null,
      highestRisk: null,
      maxUncertainty: null,
      thresholds: null,
      aggregation: null,
    });
  }, []);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      wsRef.current?.close();
    };
  }, []);

  return {
    step,
    messages,
    questions,
    report,
    riskLevel,
    riskScope,
    recommendations,
    errorMessage,
    decisionReason,
    clarificationRounds,
    reviewRequired,
    conflictCount,
    sessionId,
    reportAt,
    isLoading,
    isSending,
    specimenUrl,
    gradcamUrl,
    gradcamLoading,
    gradcamError,
    cvMetrics,
    cvCells,
    start,
    loadSession,
    cancel,
    submitClarification,
    startChat,
    closeChat,
    sendMessage,
    requestGradcam,
    reset,
  };
}