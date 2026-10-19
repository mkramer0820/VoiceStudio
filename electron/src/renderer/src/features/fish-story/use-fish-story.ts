import { useCallback, useEffect, useRef, useState } from 'react';
import { apiJson, describeError } from '@/lib/api/client';
import type { FishVoice } from '@/features/settings/fish-voice-picker';
import {
  DEFAULT_SETTINGS,
  type StoryChunk,
  type StoryDraft,
  type StorySettings,
} from './story-model';

const HISTORY_LIMIT = 40;
const SAVE_DELAY_MS = 800;

interface SavedDraft extends StoryDraft {
  id: string;
}

export interface FishDefaults {
  model: string;
  format: string;
  sample_rate: number;
  mp3_bitrate: number;
  latency: string;
  normalize: boolean;
  temperature: number | null;
  top_p: number | null;
  chunk_length: number | null;
  max_concurrency: number;
  volume: number;
  normalize_loudness: boolean;
  language: string | null;
  reference_id: string | null;
}

export interface FishBalance {
  credit: string;
  cumulative_top_up: string;
  has_free_credit: boolean | null;
}

const EMPTY: StoryDraft = { title: '', chunks: [], voices: {}, cast: {}, settings: DEFAULT_SETTINGS };

/**
 * Story-board state. Everything here is local: drafts autosave to the backend's
 * data folder and nothing is sent to Fish until the user explicitly renders,
 * loads voices, or checks the balance.
 */
export function useFishStory() {
  const [draft, setDraft] = useState<StoryDraft>(EMPTY);
  const [draftId, setDraftId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [error, setError] = useState('');
  const [defaults, setDefaults] = useState<FishDefaults | null>(null);
  const [voices, setVoices] = useState<FishVoice[] | null>(null);
  const [voicesBusy, setVoicesBusy] = useState(false);
  const [balance, setBalance] = useState<FishBalance | null>(null);
  const [balanceBusy, setBalanceBusy] = useState(false);
  const history = useRef<StoryChunk[][]>([]);
  const [canUndo, setCanUndo] = useState(false);
  const dirty = useRef(false);
  const idRef = useRef<string | null>(null);
  const draftRef = useRef(draft);
  draftRef.current = draft;

  // Load the newest saved draft and the effective defaults (both local calls).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [list, defs] = await Promise.all([
          apiJson<{ drafts: Array<{ id: string }> }>('/api/fish/story/drafts'),
          apiJson<FishDefaults>('/api/fish/story/defaults'),
        ]);
        if (!alive) return;
        setDefaults(defs);
        const newest = list.drafts[0];
        if (newest) {
          const saved = await apiJson<SavedDraft>(`/api/fish/story/drafts/${newest.id}`);
          if (!alive) return;
          idRef.current = saved.id;
          setDraftId(saved.id);
          setDraft({
            title: saved.title ?? '',
            chunks: saved.chunks ?? [],
            voices: saved.voices ?? {},
            cast: saved.cast ?? {},
            settings: { ...DEFAULT_SETTINGS, ...saved.settings },
          });
        }
      } catch (reason) {
        if (alive) setError(describeError(reason));
      } finally {
        if (alive) setReady(true);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const persist = useCallback(async () => {
    const current = draftRef.current;
    setSaveState('saving');
    try {
      const saved = idRef.current
        ? await apiJson<SavedDraft>(`/api/fish/story/drafts/${idRef.current}`, {
            method: 'PUT',
            body: JSON.stringify(current),
          })
        : await apiJson<SavedDraft>('/api/fish/story/drafts', {
            method: 'POST',
            body: JSON.stringify(current),
          });
      idRef.current = saved.id;
      setDraftId(saved.id);
      setSaveState('saved');
    } catch (reason) {
      setSaveState('error');
      setError(describeError(reason));
    }
  }, []);

  // Debounced autosave. The first render and the initial load are not edits.
  useEffect(() => {
    if (!ready || !dirty.current) return;
    const timer = setTimeout(() => {
      dirty.current = false;
      void persist();
    }, SAVE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [draft, ready, persist]);

  const edit = useCallback((next: (current: StoryDraft) => StoryDraft, undoable = true) => {
    setDraft((current) => {
      const value = next(current);
      if (value === current) return current;
      if (undoable && value.chunks !== current.chunks) {
        history.current = [...history.current.slice(-(HISTORY_LIMIT - 1)), current.chunks];
        setCanUndo(true);
      }
      dirty.current = true;
      return value;
    });
  }, []);

  const setChunks = useCallback(
    (next: StoryChunk[] | ((chunks: StoryChunk[]) => StoryChunk[]), undoable = true) =>
      edit(
        (d) => ({ ...d, chunks: typeof next === 'function' ? next(d.chunks) : next }),
        undoable,
      ),
    [edit],
  );

  const setSettings = useCallback(
    (patch: Partial<StorySettings>) =>
      edit((d) => ({ ...d, settings: { ...d.settings, ...patch } }), false),
    [edit],
  );

  const setCast = useCallback(
    (cast: Record<string, string>) => edit((d) => ({ ...d, cast }), false),
    [edit],
  );

  const setTitle = useCallback((title: string) => edit((d) => ({ ...d, title }), false), [edit]);

  const undo = useCallback(() => {
    const previous = history.current.pop();
    setCanUndo(history.current.length > 0);
    if (!previous) return;
    dirty.current = true;
    setDraft((d) => ({ ...d, chunks: previous }));
  }, []);

  const newStory = useCallback(() => {
    idRef.current = null;
    setDraftId(null);
    history.current = [];
    setCanUndo(false);
    dirty.current = false;
    setSaveState('idle');
    setDraft((d) => ({ ...EMPTY, settings: d.settings }));
  }, []);

  /** Contacts Fish (explicit button): the account's saved voices. */
  const loadVoices = useCallback(async () => {
    setVoicesBusy(true);
    setError('');
    try {
      const result = await apiJson<{ voices: FishVoice[] }>('/api/fish/voices?scope=mine');
      setVoices(result.voices);
      edit(
        (d) => ({
          ...d,
          voices: {
            ...d.voices,
            ...Object.fromEntries(result.voices.map((v) => [v.id, v.title || v.id])),
          },
        }),
        false,
      );
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setVoicesBusy(false);
    }
  }, [edit]);

  /** Contacts Fish (explicit button): remaining API credit. */
  const checkBalance = useCallback(async () => {
    setBalanceBusy(true);
    setError('');
    try {
      setBalance(await apiJson<FishBalance>('/api/fish/balance'));
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setBalanceBusy(false);
    }
  }, []);

  return {
    draft,
    draftId,
    ready,
    saveState,
    error,
    clearError: () => setError(''),
    setError,
    defaults,
    voices,
    voicesBusy,
    balance,
    balanceBusy,
    canUndo,
    setChunks,
    setSettings,
    setCast,
    setTitle,
    undo,
    newStory,
    loadVoices,
    checkBalance,
  };
}
