/** 风险等级 */
export type RiskLevel = 'low' | 'medium' | 'high' | 'emergency' | 'unreadable';

/** 细胞形态学分析结果 */
export interface Morphology {
  nuclear_cytoplasmic_ratio: number;
  nc_ratio_valid?: boolean;
  measurement_reliable?: boolean;
  area: number;
  perimeter: number;
  circularity: number;
  interpretation: string;
}

/** 单个细胞的分析结果 */
export interface CellResult {
  bbox: number[]; // [x1, y1, x2, y2]
  binary_result: { class: 'normal' | 'abnormal'; confidence: number };
  final_class: string;
  final_confidence: number;
  model_used: 'binary' | 'multiclass';
  uncertainty: number;
  probabilities: Record<string, number>;
  morphology: Morphology;
  uncertainty_metric?: string;
  uncertainty_std?: number;
}

/** CV 双模型分析结果（对应 analyze_pil_image 返回值） */
export interface CVResult {
  total_cells: number;
  abnormal_cells: number;
  cells: CellResult[];
  highest_risk: string;
  max_uncertainty: number;
  needs_clarification: boolean;
  error: string | null;
  interpretable?: boolean;
  unreadable_reason?: string | null;
  risk_level?: RiskLevel | null;
  risk_scope?: 'cell' | 'case' | null;
  uncertainty_metric?: string;
  uncertainty_aggregation?: string;
  uncertainty_thresholds?: { high: number; medium: number };
}

/** Agent 上传 / 澄清 响应 */
export interface AgentResponse {
   session_id: string;
  status: 'complete' | 'awaiting_clarification' | 'unreadable';
  risk_level?: RiskLevel | 'pending' | null;
  risk_scope?: 'cell' | 'case' | null;
  decision_reason?: string | null;
  clarification_rounds?: number; 
  recommendations?: string[];
  report?: string;
  questions?: string[];
  error_message?: string | null;
}

/** Agent HTTP 追问响应 */
export interface FollowUpResponse {
  success: boolean;
  response: string;
}

/** 健康检查响应 */
export interface HealthResponse {
  status: string;
  device: string;
  models: { yolo: string; 二分类: string; 多分类: string };
  config: {
    mc_dropout_samples: number;
    不确定阈值_high: number;
    不确定阈值_medium: number;
    异常的CONF阈值: number;
  };
}

/** 本地历史记录 */
export interface HistoryRecord {
  id: string;
  timestamp: number;
  filename: string;
  mode: 'agent' | 'quick';
  riskLevel: RiskLevel;
  totalCells?: number;
  abnormalCells?: number;
  sessionId?: string;
  reportPreview?: string;
  riskScope?: 'cell' | 'case' | null;
}

/** WebSocket 消息 */
export interface WSMessage {
  type: 'stream' | 'complete' | 'error';
  content?: string;
  message?: string;
}