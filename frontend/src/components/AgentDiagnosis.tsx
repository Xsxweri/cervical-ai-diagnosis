import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';

// API 基础配置
const API_BASE = 'http://localhost:8000';

// 类型定义
interface CVResult {
  total_cells: number;
  abnormal_cells: number;
  highest_risk: string;
  max_uncertainty: number;
  needs_clarification: boolean;
  cells: Array<{
    bbox: number[];
    final_class: string;
    final_confidence: number;
    uncertainty: number;
    morphology: {
      核细胞质比: number;
      解释: string;
    };
  }>;
}

interface DiagnosisResponse {
  session_id: string;
  status: 'complete' | 'awaiting_clarification';
  report?: string;
  questions?: string[];
  recommendations?: string[];
  cv_results?: CVResult;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  type?: 'text' | 'image' | 'report';
}

const AgentDiagnosis: React.FC = () => {
  // 状态
  const [step, setStep] = useState<'upload' | 'analyzing' | 'clarification' | 'report' | 'chat'>('upload');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [questions, setQuestions] = useState<string[]>([]);
  const [report, setReport] = useState<string>('');
  const [cvResults, setCvResults] = useState<CVResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // 自动滚动到底部
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // 上传图像并开始诊断
  const handleUpload = async (file: File) => {
    setStep('analyzing');
    setIsLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<DiagnosisResponse>(
        `${API_BASE}/api/agent/upload`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 120000 // 2分钟，模型推理可能较慢
        }
      );

      const data = response.data;
      setSessionId(data.session_id);
      setCvResults(data.cv_results || null);

      if (data.status === 'awaiting_clarification' && data.questions) {
        // 需要澄清
        setStep('clarification');
        setQuestions(data.questions);
        setMessages([
          { role: 'system', content: `AI检测到诊断不确定性较高，需要补充信息...` },
          ...data.questions.map((q, i) => ({
            role: 'assistant' as const,
            content: `${i + 1}. ${q}`
          }))
        ]);
      } else {
        // 直接生成报告
        setStep('report');
        setReport(data.report || '');
        setMessages([
          { role: 'system', content: '诊断完成' },
          { role: 'assistant', content: data.report || '', type: 'report' }
        ]);
        
        // 连接 WebSocket 用于后续追问
        connectWebSocket(data.session_id);
      }
    } catch (error) {
      setStep('upload');
      alert(`诊断失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 提交澄清回答
  const submitClarification = async (answers: string[]) => {
    if (!sessionId) return;
    
    setIsLoading(true);
    const answerText = answers.join('；');

    try {
      const response = await axios.post<DiagnosisResponse>(
        `${API_BASE}/api/agent/clarify/${sessionId}`,
        { user_input: answerText }
      );

      const data = response.data;
      setStep('report');
      setReport(data.report || '');
      setMessages(prev => [
        ...prev,
        { role: 'user', content: answerText },
        { role: 'assistant', content: data.report || '', type: 'report' }
      ]);
      
      connectWebSocket(sessionId);
    } catch (error) {
      alert(`澄清处理失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 连接 WebSocket
  const connectWebSocket = useCallback((sid: string) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/agent/chat/${sid}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setStep('chat');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'stream') {
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last?.role === 'assistant' && last.type === 'text') {
            // 追加到现有消息
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + data.content }
            ];
          }
          // 新消息
          return [...prev, { role: 'assistant', content: data.content, type: 'text' }];
        });
      } else if (data.type === 'complete') {
        // 流式完成
      } else if (data.type === 'error') {
        setMessages(prev => [...prev, { 
          role: 'system', 
          content: `错误: ${data.message}` 
        }]);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    wsRef.current = ws;
  }, []);

  // 发送追问
  const sendFollowUp = (input: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      // 降级为 HTTP
      sendFollowUpHttp(input);
      return;
    }
    
    setMessages(prev => [...prev, { role: 'user', content: input }]);
    
    wsRef.current.send(JSON.stringify({ message: input }));
  };

  // HTTP 降级追问
  const sendFollowUpHttp = async (input: string) => {
    if (!sessionId) return;
    
    try {
      const response = await axios.post(`${API_BASE}/api/agent/followup`, {
        session_id: sessionId,
        message: input
      });
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.data.response,
        type: 'text'
      }]);
    } catch (error) {
      alert(`追问失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  };

  // 渲染上传界面
  const renderUpload = () => (
    <div className="flex flex-col items-center justify-center py-20">
      <div 
        className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-500 transition-colors cursor-pointer"
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="text-6xl mb-4">📤</div>
        <h3 className="text-xl font-semibold mb-2">上传宫颈细胞图像</h3>
        <p className="text-gray-500 mb-4">支持 JPG、PNG、BMP 格式，最大 30MB</p>
        <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700">
          选择文件
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleUpload(file);
          }}
        />
      </div>
      
      <div className="mt-8 grid grid-cols-3 gap-4 text-center text-sm text-gray-600">
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-2xl mb-2">🔬</div>
          <div>YOLOv8 细胞检测</div>
          <div className="text-xs text-gray-400">mAP 97.4%</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-2xl mb-2">🧠</div>
          <div>ResNet50 分类</div>
          <div className="text-xs text-gray-400">准确率 96%</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-2xl mb-2">🤖</div>
          <div>LangGraph Agent</div>
          <div className="text-xs text-gray-400">智能追问</div>
        </div>
      </div>
    </div>
  );

  // 渲染分析中
  const renderAnalyzing = () => (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mb-4"></div>
      <h3 className="text-xl font-semibold">AI 正在分析细胞图像...</h3>
      <p className="text-gray-500 mt-2">YOLO检测 → ResNet分类 → MC Dropout不确定性评估</p>
      {isLoading && (
        <div className="mt-4 w-64 bg-gray-200 rounded-full h-2">
          <div className="bg-blue-600 h-2 rounded-full animate-pulse w-full"></div>
        </div>
      )}
    </div>
  );

  // 渲染澄清问题
  const renderClarification = () => {
    const [answers, setAnswers] = useState<string[]>(new Array(questions.length).fill(''));
    
    return (
      <div className="max-w-2xl mx-auto py-10">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold text-yellow-800 mb-2">
            ⚠️ 诊断不确定性较高
          </h3>
          <p className="text-yellow-700">
            AI 检测到部分细胞特征不明确，需要补充临床信息以提高诊断准确性。
          </p>
        </div>
        
        <div className="space-y-4">
          {questions.map((q, i) => (
            <div key={i} className="bg-white rounded-lg shadow p-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {i + 1}. {q}
              </label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="请输入..."
                value={answers[i]}
                onChange={(e) => {
                  const newAnswers = [...answers];
                  newAnswers[i] = e.target.value;
                  setAnswers(newAnswers);
                }}
              />
            </div>
          ))}
        </div>
        
        <button
          className="mt-6 w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
          disabled={answers.some(a => !a.trim()) || isLoading}
          onClick={() => submitClarification(answers)}
        >
          {isLoading ? '重新分析中...' : '提交并重新诊断'}
        </button>
      </div>
    );
  };

  // 渲染报告
  const renderReport = () => (
    <div className="max-w-4xl mx-auto">
      {/* CV 结果摘要 */}
      {cvResults && (
        <div className="bg-white rounded-lg shadow mb-6 p-6">
          <h3 className="text-lg font-semibold mb-4">📊 细胞检测摘要</h3>
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div className="text-center p-3 bg-gray-50 rounded">
              <div className="text-2xl font-bold text-blue-600">{cvResults.total_cells}</div>
              <div className="text-sm text-gray-500">检测细胞数</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded">
              <div className="text-2xl font-bold text-red-600">{cvResults.abnormal_cells}</div>
              <div className="text-sm text-gray-500">异常细胞</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded">
              <div className="text-2xl font-bold text-orange-600">
                {(cvResults.max_uncertainty * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">不确定性</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded">
              <div className="text-2xl font-bold text-purple-600">{cvResults.highest_risk}</div>
              <div className="text-sm text-gray-500">最高风险</div>
            </div>
          </div>
          
          {/* 细胞详情表格 */}
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left">类别</th>
                <th className="px-4 py-2 text-left">置信度</th>
                <th className="px-4 py-2 text-left">不确定性</th>
                <th className="px-4 py-2 text-left">核质比</th>
                <th className="px-4 py-2 text-left">形态解读</th>
              </tr>
            </thead>
            <tbody>
              {cvResults.cells.slice(0, 5).map((cell, i) => (
                <tr key={i} className="border-t">
                  <td className="px-4 py-2">{cell.final_class}</td>
                  <td className="px-4 py-2">{(cell.final_confidence * 100).toFixed(1)}%</td>
                  <td className="px-4 py-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      cell.uncertainty > 0.3 ? 'bg-red-100 text-red-800' :
                      cell.uncertainty > 0.15 ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {(cell.uncertainty * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-2">{cell.morphology.核细胞质比}</td>
                  <td className="px-4 py-2 text-gray-600">{cell.morphology.解释}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      {/* 诊断报告 */}
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">📝 AI 诊断报告</h3>
        <div className="prose max-w-none whitespace-pre-wrap">{report}</div>
      </div>
      
      {/* 进入追问 */}
      <button
        className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700"
        onClick={() => setStep('chat')}
      >
        进入医生追问模式 💬
      </button>
    </div>
  );

  // 渲染对话
  const renderChat = () => (
    <div className="max-w-4xl mx-auto h-[600px] flex flex-col">
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 p-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : msg.role === 'system'
                  ? 'bg-gray-100 text-gray-600 text-sm'
                  : msg.type === 'report'
                  ? 'bg-green-50 border border-green-200 text-gray-800'
                  : 'bg-white border border-gray-200 text-gray-800'
              }`}
            >
              {msg.type === 'report' ? (
                <div className="whitespace-pre-wrap">{msg.content}</div>
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>
      
      <div className="border-t p-4 bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
            placeholder="输入追问内容..."
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                const input = e.currentTarget;
                if (input.value.trim()) {
                  sendFollowUp(input.value);
                  input.value = '';
                }
              }
            }}
          />
          <button
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
            onClick={() => {
              const input = document.querySelector('input') as HTMLInputElement;
              if (input?.value.trim()) {
                sendFollowUp(input.value);
                input.value = '';
              }
            }}
          >
            发送
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div>
      {step === 'upload' && renderUpload()}
      {step === 'analyzing' && renderAnalyzing()}
      {step === 'clarification' && renderClarification()}
      {step === 'report' && renderReport()}
      {step === 'chat' && renderChat()}
    </div>
  );
};

export default AgentDiagnosis;