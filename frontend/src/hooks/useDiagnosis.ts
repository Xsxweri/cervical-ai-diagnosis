import { useCallback, useEffect, useRef, useState } from 'react';
import type { AgentResponse, RiskLevel } from '../types';
import { agentClarify, agentFollowUp, agentUpload, connectAgentWS } from '../libts/api';
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
    (res: AgentResponse, filename: string) => {
      const text = res.report ?? '';
      setReport(text);
      setRecommendations(res.recommendations ?? []);
      setRiskScope(res.risk_scope ?? null);

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
      setStep('report');
      pushMessage({ role: 'assistant', content: text, type: 'report' });
      addHistory({
        id: res.session_id,
        timestamp: Date.now(),
        filename,
        mode: 'agent',
        riskLevel: finalRisk,
        riskScope: res.risk_scope ?? null,
        sessionId: res.session_id,
        reportPreview: text.slice(0, 200),
      });
      connectWS(res.session_id);
    },
    [connectWS, pushMessage]
  );

  const start = useCallback(
    async (file: File) => {
      filenameRef.current = file.name;
      setStep('analyzing');
      setIsLoading(true);
      setErrorMessage(null);
      setRiskScope(null);
      setRecommendations([]);
      pushMessage({ role: 'system', content: `已上传图像：${file.name}` });
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const res = await agentUpload(file, controller.signal);
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
    isLoading,
    isSending,
    start,
    cancel,
    submitClarification,
    startChat,
    sendMessage,
    reset,
  };
}