import type { HistoryRecord } from '../types';

const KEY = 'cervical-diagnosis-history';
const MAX = 50;

export function getHistory(): HistoryRecord[] {
  try {
    const raw = localStorage.getItem(KEY);
    const list = raw ? (JSON.parse(raw) as HistoryRecord[]) : [];
    return Array.isArray(list) ? list.sort((a, b) => b.timestamp - a.timestamp) : [];
  } catch {
    return [];
  }
}

export function addHistory(record: HistoryRecord): HistoryRecord[] {
  const list = [record, ...getHistory().filter((r) => r.id !== record.id)].slice(0, MAX);
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    /* 存储失败静默 */
  }
  return list;
}

export function removeHistory(id: string): HistoryRecord[] {
  const list = getHistory().filter((r) => r.id !== id);
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    /* noop */
  }
  return list;
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* noop */
  }
}