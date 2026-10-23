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
  opus_bitrate: number | null;
  repetition_penalty: number | null;
  max_new_tokens: number | null;
  min_chunk_length: number | null;
  condition_on_previous_chunks: boolean | null;
  early_stop_threshold: number | null;
  max_concurrency: number;
  volume: number;
  normalize_loudness: boolean;
  language: string | null;
  reference_id: string | null;
  output_dir: string | null;
  output_dir_default: string;
  output_dir_effective: string;
}

/** A voice you have saved, named or used: your character name beside what Fish says about it. */
export interface VoiceRecord {
  voice_id: string;
  alias: string;
  notes: string;
  fish_name: string;
  description: string;
  languages: string[];
  tags: string[];
  visibility: string | null;
  state: string | null;
  author: string | null;
  fish_url: string;
  source: string;
  added_at: number;
  last_used_at: number | null;
  uses: number;
  lines: number;
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
  const [library, setLibrary] = useState<VoiceRecord[]>([]);
  const libraryRef = useRef<VoiceRecord[]>([]);
  libraryRef.current = library;
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

  /** Local: the voices remembered from earlier lists, additions and renders. */
  const refreshLibrary = useCallback(async () => {
    try {
      const result = await apiJson<{ voices?: VoiceRecord[] }>('/api/fish/library');
      setLibrary(result.voices ?? []);
    } catch {
      // The library is a convenience; the board works without it.
    }
  }, []);

  useEffect(() => {
    void refreshLibrary();
  }, [refreshLibrary]);

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

  /**
   * Local only: put a finished run's saved story (lines, voices, settings) on the board as a new
   * draft, so it can be re-rendered with different settings. The previous draft stays saved.
   */
  const loadRun = useCallback(async (jobId: string): Promise<{ title: string; lines: number } | null> => {
    setError('');
    try {
      const run = await apiJson<{ draft: StoryDraft }>(`/api/fish/story/jobs/${jobId}/load`);
      idRef.current = null;
      setDraftId(null);
      history.current = [];
      setCanUndo(false);
      dirty.current = true;
      setDraft({
        title: run.draft.title ?? '',
        chunks: run.draft.chunks ?? [],
        voices: run.draft.voices ?? {},
        cast: run.draft.cast ?? {},
        settings: { ...DEFAULT_SETTINGS, ...run.draft.settings },
      });
      return { title: run.draft.title ?? '', lines: run.draft.chunks?.length ?? 0 };
    } catch (reason) {
      setError(describeError(reason));
      return null;
    }
  }, []);

  /** Saves render defaults (and the output folder) to Settings; local only. */
  const saveDefaults = useCallback(async (patch: Record<string, unknown>): Promise<boolean> => {
    setError('');
    try {
      setDefaults(
        await apiJson<FishDefaults>('/api/fish/settings', {
          method: 'PUT',
          body: JSON.stringify(patch),
        }),
      );
      return true;
    } catch (reason) {
      setError(describeError(reason));
      return false;
    }
  }, []);

  /** Contacts Fish (explicit button): the account's saved voices. */
  const loadVoices = useCallback(async () => {
    setVoicesBusy(true);
    setError('');
    try {
      const result = await apiJson<{ voices: FishVoice[] }>('/api/fish/voices?scope=mine');
      setVoices(result.voices);
      const alias = new Map(libraryRef.current.map((r) => [r.voice_id, r.alias]));
      edit(
        (d) => ({
          ...d,
          voices: {
            // Names already on the board (and your character names) win over Fish's titles.
            ...Object.fromEntries(
              result.voices.map((v) => [v.id, alias.get(v.id) || v.title || v.id]),
            ),
            ...d.voices,
          },
        }),
        false,
      );
      void refreshLibrary(); // the server remembers your account's voices as they are listed
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setVoicesBusy(false);
    }
  }, [edit, refreshLibrary]);

  /** Local: name a character, add notes, or store Fish's details for a voice in the library. */
  const saveVoice = useCallback(
    async (
      id: string,
      patch: { alias?: string; notes?: string; fish?: Partial<FishVoice> },
    ): Promise<VoiceRecord | null> => {
      try {
        const record = await apiJson<VoiceRecord>(`/api/fish/library/${encodeURIComponent(id)}`, {
          method: 'PUT',
          body: JSON.stringify(patch),
        });
        if (record?.voice_id) {
          setLibrary((rows) => [record, ...rows.filter((r) => r.voice_id !== id)]);
          return record;
        }
      } catch (reason) {
        setError(describeError(reason));
      }
      return null;
    },
    [],
  );

  /**
   * Local only: put a voice on the board by its Fish id (public voices, a friend's, one you pasted).
   * ``name`` is a character name you typed; Fish's own title is kept beside it in the library.
   */
  const addVoice = useCallback(
    (id: string, options: { name?: string; fish?: FishVoice } = {}) => {
      const known = libraryRef.current.find((r) => r.voice_id === id);
      const typed = options.name?.trim();
      edit(
        (d) => ({
          ...d,
          voices: {
            ...d.voices,
            [id]: typed || d.voices[id] || known?.alias || options.fish?.title || known?.fish_name || id,
          },
        }),
        false,
      );
      void saveVoice(id, { ...(typed ? { alias: typed } : {}), ...(options.fish ? { fish: options.fish } : {}) });
    },
    [edit, saveVoice],
  );

  /** Give a character its own name everywhere: in the library and on this board. */
  const renameVoice = useCallback(
    async (id: string, alias: string) => {
      const record = await saveVoice(id, { alias });
      const shown = alias.trim() || record?.fish_name || id;
      edit((d) => (id in d.voices ? { ...d, voices: { ...d.voices, [id]: shown } } : d), false);
    },
    [edit, saveVoice],
  );

  const forgetVoice = useCallback(async (id: string) => {
    try {
      await apiJson(`/api/fish/library/${encodeURIComponent(id)}`, { method: 'DELETE' });
      setLibrary((rows) => rows.filter((r) => r.voice_id !== id));
    } catch (reason) {
      setError(describeError(reason));
    }
  }, []);

  /** Contacts Fish (explicit action): search the public Voice Library. */
  const searchLibrary = useCallback(async (title: string, language: string): Promise<FishVoice[] | null> => {
    setError('');
    try {
      const q = new URLSearchParams({ scope: 'library' });
      if (title.trim()) q.set('title', title.trim());
      if (language.trim()) q.set('language', language.trim());
      return (await apiJson<{ voices: FishVoice[] }>(`/api/fish/voices?${q}`)).voices;
    } catch (reason) {
      setError(describeError(reason));
      return null;
    }
  }, []);

  /** Contacts fish.audio's website (explicit action): the voices in a public collection. */
  const importCollection = useCallback(
    async (url: string): Promise<{ title: string; voices: FishVoice[] } | null> => {
      setError('');
      try {
        return await apiJson<{ title: string; voices: FishVoice[] }>('/api/fish/collection', {
          method: 'POST',
          body: JSON.stringify({ url }),
        });
      } catch (reason) {
        setError(describeError(reason));
        return null;
      }
    },
    [],
  );

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
    loadRun,
    loadVoices,
    library,
    refreshLibrary,
    saveVoice,
    renameVoice,
    forgetVoice,
    addVoice,
    searchLibrary,
    importCollection,
    checkBalance,
    saveDefaults,
  };
}
