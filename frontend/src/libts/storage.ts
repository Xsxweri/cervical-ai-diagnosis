import type { HistoryRecord } from '../types';

const KEY = 'cervical-diagnosis-history';
const TOKEN_KEY = 'cervical-auth-token';
const MAX = 50;

const DOCTOR_KEY = 'cervical-doctor-profile';

export interface DoctorProfile {
  name?: string;
  department?: string;
  hospital?: string;
}

export function getDoctor(): DoctorProfile | null {
  try {
    const raw = localStorage.getItem(DOCTOR_KEY);
    return raw ? (JSON.parse(raw) as DoctorProfile) : null;
  } catch {
    return null;
  }
}

export function setDoctor(doctor: DoctorProfile | null): void {
  try {
    if (doctor) localStorage.setItem(DOCTOR_KEY, JSON.stringify(doctor));
  } catch {
    /* noop */
  }
}

export function clearDoctor(): void {
  try {
    localStorage.removeItem(DOCTOR_KEY);
  } catch {
    /* noop */
  }
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* noop */
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* noop */
  }
}

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