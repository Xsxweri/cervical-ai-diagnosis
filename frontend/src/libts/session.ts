/**
 * 当前账号会话
**/
import { useEffect, useState } from 'react';
import { fetchMe } from './api';
import { getCachedMe, setCachedMe } from './storage';
import type { UserPublic } from '../types';

const TTL = 5 * 60 * 1000;

let cached: { me: UserPublic; at: number } | null = null;
let inflight: Promise<UserPublic> | null = null;

function fresh(): UserPublic | null {
  if (cached && Date.now() - cached.at < TTL) return cached.me;
  return getCachedMe();      // 首屏用 localStorage，后续会刷新
}

/** 全应用共享一次请求；force=true 可强制刷新（如保存账号后） */
export function loadMe(force = false): Promise<UserPublic> {
  if (!force && cached && Date.now() - cached.at < TTL) {
    return Promise.resolve(cached.me);
  }
  if (inflight) return inflight;

  inflight = fetchMe()
    .then((me) => {
      cached = { me, at: Date.now() };
      setCachedMe(me);
      return me;
    })
    .finally(() => {
      inflight = null;
    });

  return inflight;
}

interface MeState {
  me: UserPublic | null;
  loading: boolean;
  error: string | null;
  reload: (force?: boolean) => void;
}

export function useMe(): MeState {
  const [me, setMe] = useState<UserPublic | null>(() => fresh());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    loadMe(tick > 0)
      .then((u) => {
        if (!alive) return;
        setMe(u);
        setError(null);
      })
      .catch((e: unknown) => {
        if (alive) setError(errText(e, '账号信息读取失败'));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [tick]);

  return {
    me,
    loading,
    error,
    reload: (force = true) => setTick((n) => n + (force ? 1 : 0)),
  };
}

/** FastAPI 错误体是 { detail: string }；detail 也可能是校验错误数组 */
export function errText(e: unknown, fallback: string): string {
  const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: string };
    if (first?.msg) return first.msg;
  }
  return fallback;
}