import { useCallback, useEffect, useRef, useState } from 'react';
import type { AgentResponse } from '../types';
import { agentClarify, agentFollowUp, agentUpload, connectAgentWS } from '../libts/api';
import { riskFromText } from '../libts/format';
import { addHistory } from '../libts/storage';

export type DiagnosisStep = 'upload' | 'analyzing' | 'clarification' | 'report' | 'chat';

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
  const wsRef = useRef<WebSocket | null>(null);
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
      setStep('report');
      pushMessage({ role: 'assistant', content: text, type: 'report' });
      addHistory({
        id: res.session_id,
        timestamp: Date.now(),
        filename,
        mode: 'agent',
        riskLevel: riskFromText(text),
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
      pushMessage({ role: 'system', content: `已上传图像：${file.name}` });
      try {
        const res = await agentUpload(file);
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
        pushMessage({
          role: 'system',
          content: `诊断失败：${e instanceof Error ? e.message : '未知错误'}`,
        });
        setStep('upload');
      } finally {
        setIsLoading(false);
      }
    },
    [finishReport, pushMessage]
  );

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
      if (!sessionId) return;
      pushMessage({ role: 'user', content: input });
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ message: input }));
      } else {
        try {
          const resp = await agentFollowUp(sessionId, input);
          pushMessage({ role: 'assistant', content: resp, type: 'text' });
        } catch (e) {
          pushMessage({
            role: 'system',
            content: `追问失败：${e instanceof Error ? e.message : '未知错误'}`,
          });
        }
      }
    },
    [sessionId, pushMessage]
  );

  const reset = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setStep('upload');
    setMessages([]);
    setSessionId(null);
    setQuestions([]);
    setReport('');
  }, []);

  useEffect(() => {
    return () => wsRef.current?.close();
  }, []);

  return {
    step,
    messages,
    questions,
    report,
    isLoading,
    start,
    submitClarification,
    startChat,
    sendMessage,
    reset,
  };
}