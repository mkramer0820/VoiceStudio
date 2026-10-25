import { useCallback, useEffect, useRef, useState } from 'react';
import { apiJson, describeError } from '@/lib/api/client';
import {
  buildRunRequest,
  duplicateTake,
  isRunnable,
  newTake,
  type JobView,
  type QwenSettings,
  type QwenState,
  type SessionRow,
  type TakeDraft,
  type TakeView,
} from './qwen-model';

const POLL_MS = 1000;
const STORE_KEY = 'qwen-playground:v1';

interface Stored {
  takes: TakeDraft[];
  sessionName: string;
  groupByModel: boolean;
}

// Per-viewer convenience only: the page works without it.
function readStored(): Stored | null {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    const parsed = raw ? (JSON.parse(raw) as Stored) : null;
    return parsed && Array.isArray(parsed.takes) ? parsed : null;
  } catch {
    return null;
  }
}

function writeStored(value: Stored) {
  try {
    window.localStorage.setItem(STORE_KEY, JSON.stringify(value));
  } catch {
    // private window or blocked storage
  }
}

export function useQwenPlayground() {
  const [state, setState] = useState<QwenState | null>(null);
  const [error, setError] = useState('');
  const [takes, setTakes] = useState<TakeDraft[]>(() => readStored()?.takes ?? []);
  const [sessionName, setSessionName] = useState(() => readStored()?.sessionName ?? 'playground');
  const [groupByModel, setGroupByModel] = useState(() => readStored()?.groupByModel ?? true);
  const [outputDir, setOutputDir] = useState('');
  const [warmupRuns, setWarmupRuns] = useState(1);
  const [warmupText, setWarmupText] = useState('');
  const [job, setJob] = useState<JobView | null>(null);
  const [busy, setBusy] = useState(false);
  const stateRef = useRef<QwenState | null>(null);
  stateRef.current = state;

  const applyState = useCallback((next: QwenState, first = false) => {
    setState(next);
    if (first) {
      setOutputDir(next.settings.output_dir || '');
      setWarmupRuns(next.settings.warmup_runs ?? 1);
      setWarmupText(next.settings.warmup_text ?? '');
      setTakes((cur) => (cur.length ? cur : [newTake(next.models)]));
    }
  }, []);

  const refresh = useCallback(async (first = false) => {
    try {
      applyState(await apiJson<QwenState>('/api/qwen/state'), first);
      return true;
    } catch (reason) {
      setError(describeError(reason));
      return false;
    }
  }, [applyState]);

  useEffect(() => {
    void refresh(true);
  }, [refresh]);

  useEffect(() => writeStored({ takes, sessionName, groupByModel }), [takes, sessionName, groupByModel]);

  // Poll the running job; there is no websocket.
  const jobId = job?.id;
  const jobRunning = job?.status === 'running';
  useEffect(() => {
    if (!jobId || !jobRunning) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const next = await apiJson<JobView>(`/api/qwen/playground/jobs/${jobId}`);
        if (!alive) return;
        setJob(next);
        if (next.status !== 'running') return;
      } catch (reason) {
        if (!alive) return;
        setError(describeError(reason));
        return;
      }
      timer = setTimeout(tick, POLL_MS);
    };
    timer = setTimeout(tick, POLL_MS);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [jobId, jobRunning]);

  const saveSettings = useCallback(async (patch: Partial<QwenSettings>) => {
    try {
      await apiJson('/api/qwen/settings', { method: 'PUT', body: JSON.stringify(patch) });
      return await refresh();
    } catch (reason) {
      setError(describeError(reason));
      return false;
    }
  }, [refresh]);

  // ── takes ─────────────────────────────────────────────────────────────
  const models = state?.models ?? [];
  const addTake = useCallback(() => setTakes((cur) => [...cur, newTake(stateRef.current?.models ?? [])]), []);
  const updateTake = useCallback(
    (id: string, change: (take: TakeDraft) => TakeDraft) =>
      setTakes((cur) => cur.map((t) => (t.id === id ? change(t) : t))),
    [],
  );
  const removeTake = useCallback((id: string) => setTakes((cur) => cur.filter((t) => t.id !== id)), []);
  const copyTake = useCallback(
    (id: string) =>
      setTakes((cur) => {
        const at = cur.findIndex((t) => t.id === id);
        return at < 0 ? cur : [...cur.slice(0, at + 1), duplicateTake(cur[at]), ...cur.slice(at + 1)];
      }),
    [],
  );
  const moveTake = useCallback(
    (id: string, by: -1 | 1) =>
      setTakes((cur) => {
        const at = cur.findIndex((t) => t.id === id);
        const to = at + by;
        if (at < 0 || to < 0 || to >= cur.length) return cur;
        const next = [...cur];
        [next[at], next[to]] = [next[to], next[at]];
        return next;
      }),
    [],
  );

  // ── runs ──────────────────────────────────────────────────────────────
  /** Run every take, or only `only`. Takes without a model or text are left out. */
  const run = useCallback(
    async (only?: ReadonlySet<string>) => {
      const st = stateRef.current;
      if (!st || busy) return false;
      const profiles = st.profiles;
      const chosen = takes
        .filter((t) => !only || only.has(t.id))
        .filter((t) => isRunnable(t, st.models.find((m) => m.id === t.modelId), profiles));
      if (!chosen.length) return false;
      setBusy(true);
      setError('');
      try {
        // Session routes (notes, audio, past sessions) read the saved folder, so save it first.
        const wanted = outputDir.trim();
        if (wanted && wanted !== st.settings.output_dir) {
          await apiJson('/api/qwen/settings', { method: 'PUT', body: JSON.stringify({ output_dir: wanted }) });
        }
        const body = buildRunRequest(chosen, { sessionName, groupByModel, warmupRuns, warmupText });
        setJob(await apiJson<JobView>('/api/qwen/playground/run', { method: 'POST', body: JSON.stringify(body) }));
        return true;
      } catch (reason) {
        setError(describeError(reason));
        return false;
      } finally {
        setBusy(false);
      }
    },
    [busy, takes, outputDir, sessionName, groupByModel, warmupRuns, warmupText],
  );

  const cancel = useCallback(async () => {
    if (!jobId) return;
    try {
      await apiJson(`/api/qwen/playground/jobs/${jobId}/cancel`, { method: 'POST' });
    } catch (reason) {
      setError(describeError(reason));
    }
  }, [jobId]);

  const unload = useCallback(async () => {
    try {
      await apiJson('/api/qwen/playground/unload', { method: 'POST' });
    } catch (reason) {
      setError(describeError(reason));
    }
  }, []);

  /** Save notes or a star into the manifest, then mirror it in the local copy. */
  const annotate = useCallback(
    async (session: string, takeId: string, patch: { notes?: string; starred?: boolean }) => {
      try {
        await apiJson(`/api/qwen/playground/sessions/${encodeURIComponent(session)}/takes/${encodeURIComponent(takeId)}`, {
          method: 'PATCH',
          body: JSON.stringify(patch),
        });
        setJob((cur) =>
          cur && cur.session === session
            ? { ...cur, takes: cur.takes.map((t): TakeView => (t.id === takeId ? { ...t, ...patch } : t)) }
            : cur,
        );
        return true;
      } catch (reason) {
        setError(describeError(reason));
        return false;
      }
    },
    [],
  );

  const addProfile = useCallback(
    async (form: FormData) => {
      const created = await apiJson<{ id: string }>('/profiles', { method: 'POST', body: form });
      await refresh();
      return created.id;
    },
    [refresh],
  );

  return {
    state, models, error, setError, takes, setTakes, sessionName, setSessionName, groupByModel, setGroupByModel,
    outputDir, setOutputDir, warmupRuns, setWarmupRuns, warmupText, setWarmupText, job, busy, jobRunning,
    refresh, saveSettings, addTake, updateTake, removeTake, copyTake, moveTake, run, cancel, unload, annotate, addProfile,
  };
}

export async function listSessions(): Promise<SessionRow[]> {
  return (await apiJson<{ sessions: SessionRow[] }>('/api/qwen/playground/sessions')).sessions;
}

export function readSession(session: string) {
  return apiJson<{ name?: string; session?: string; takes: TakeView[]; status?: string; folder?: string }>(
    `/api/qwen/playground/sessions/${encodeURIComponent(session)}`,
  );
}
