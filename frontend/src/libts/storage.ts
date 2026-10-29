import type { HistoryRecord,UserPublic } from '../types';

const KEY = 'cervical-diagnosis-history';
const TOKEN_KEY = 'cervical-auth-token';
const MAX = 50;

const DOCTOR_KEY = 'cervical-doctor-profile';
const ME_KEY = 'cervical-me';

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

export function getCachedMe(): UserPublic | null {
  try {
    const raw = localStorage.getItem(ME_KEY);
    return raw ? (JSON.parse(raw) as UserPublic) : null;
  } catch {
    return null;
  }
}

export function setCachedMe(me: UserPublic | null): void {
  try {
    if (me) localStorage.setItem(ME_KEY, JSON.stringify(me));
    else localStorage.removeItem(ME_KEY);
  } catch {
    /* noop */
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
    localStorage.removeItem(ME_KEY); 
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