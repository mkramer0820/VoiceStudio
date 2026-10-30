import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, apiJson, describeError } from '@/lib/api/client';
import type { QwenState } from '../qwen-playground/qwen-model';
import {
  DEFAULT_STORY_SETTINGS,
  profileBody,
  type Draft,
  type Emotion,
  type EmotionCatalog,
  type Profile,
  type ProfileForm,
  type StoryChunk,
  type StoryJob,
  type StorySettings,
} from './story-model';

const POLL_MS = 1000;
const SAVE_DELAY_MS = 800;

interface StoryRow {
  id: string;
  title: string;
  chunks: number;
}

interface Parsed {
  chunks: StoryChunk[];
}

export function useQwenStory() {
  const [state, setState] = useState<QwenState | null>(null);
  const [catalog, setCatalog] = useState<EmotionCatalog | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [stories, setStories] = useState<StoryRow[]>([]);
  const [error, setError] = useState('');
  const [storyId, setStoryId] = useState<string | null>(null);
  const [title, setTitle] = useState('Untitled story');
  const [script, setScript] = useState('');
  const [chunks, setChunks] = useState<StoryChunk[]>([]);
  const [narrator, setNarrator] = useState<string | null>(null);
  const [cast, setCast] = useState<Record<string, string>>({});
  const [emotions, setEmotions] = useState<Record<string, Emotion>>({});
  const [settings, setSettings] = useState<StorySettings>(DEFAULT_STORY_SETTINGS);
  const [job, setJob] = useState<StoryJob | null>(null);
  const [busy, setBusy] = useState(false);
  const dirty = useRef(false);

  const draft: Draft = { title, chunks, narrator_profile_id: narrator, cast, emotions, settings };
  const draftRef = useRef(draft);
  draftRef.current = draft;

  const fail = useCallback((reason: unknown) => {
    setError(describeError(reason));
    return false;
  }, []);

  const loadProfiles = useCallback(async () => {
    try {
      setProfiles((await apiJson<{ profiles: Profile[] }>('/api/qwen/profiles')).profiles);
    } catch (reason) {
      fail(reason);
    }
  }, [fail]);

  const loadStories = useCallback(async () => {
    try {
      setStories((await apiJson<{ stories: StoryRow[] }>('/api/qwen/story/drafts')).stories);
    } catch (reason) {
      fail(reason);
    }
  }, [fail]);

  useEffect(() => {
    void (async () => {
      try {
        setState(await apiJson<QwenState>('/api/qwen/state'));
        setCatalog(await apiJson<EmotionCatalog>('/api/qwen/emotions'));
      } catch (reason) {
        fail(reason);
      }
      await loadProfiles();
      await loadStories();
    })();
  }, [fail, loadProfiles, loadStories]);

  // ── draft ──────────────────────────────────────────────────────────────
  const edit = useCallback(<T,>(set: (value: T) => void, value: T) => {
    dirty.current = true;
    set(value);
  }, []);

  const save = useCallback(async (): Promise<string | null> => {
    try {
      const body = JSON.stringify(draftRef.current);
      const path = storyId ? `/api/qwen/story/drafts/${storyId}` : '/api/qwen/story/drafts';
      const saved = await apiJson<{ id: string }>(path, { method: storyId ? 'PUT' : 'POST', body });
      dirty.current = false;
      setStoryId(saved.id);
      void loadStories();
      return saved.id;
    } catch (reason) {
      fail(reason);
      return null;
    }
  }, [storyId, fail, loadStories]);

  // Mappings are kept per story: save a moment after the last change.
  useEffect(() => {
    if (!storyId || !dirty.current) return;
    const timer = setTimeout(() => void save(), SAVE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [storyId, title, chunks, narrator, cast, emotions, settings, save]);

  const parse = useCallback(async () => {
    if (!script.trim()) return;
    setBusy(true);
    try {
      const parsed = await apiJson<Parsed>('/api/qwen/story/parse', { method: 'POST', body: JSON.stringify({ text: script }) });
      edit(setChunks, parsed.chunks);
      setJob(null);
    } catch (reason) {
      fail(reason);
    } finally {
      setBusy(false);
    }
  }, [script, edit, fail]);

  const open = useCallback(
    async (id: string) => {
      try {
        const d = await apiJson<Draft & { id: string }>(`/api/qwen/story/drafts/${id}`);
        dirty.current = false;
        setStoryId(d.id);
        setTitle(d.title);
        setChunks(d.chunks);
        setNarrator(d.narrator_profile_id);
        setCast(d.cast);
        setEmotions(d.emotions);
        setSettings({ ...DEFAULT_STORY_SETTINGS, ...d.settings });
        setJob(null);
      } catch (reason) {
        fail(reason);
      }
    },
    [fail],
  );

  const remove = useCallback(
    async (id: string) => {
      try {
        await apiJson(`/api/qwen/story/drafts/${id}`, { method: 'DELETE' });
        if (id === storyId) setStoryId(null);
        await loadStories();
      } catch (reason) {
        fail(reason);
      }
    },
    [storyId, fail, loadStories],
  );

  // ── render ─────────────────────────────────────────────────────────────
  const render = useCallback(
    async (chunkIds?: string[]) => {
      setError('');
      const id = await save();
      if (!id) return;
      const run = chunkIds && job ? job.run : undefined;
      try {
        const next = await apiJson<StoryJob>('/api/qwen/story/render', {
          method: 'POST',
          body: JSON.stringify({ story_id: id, draft: draftRef.current, chunk_ids: chunkIds ?? null, run: run ?? null }),
        });
        setJob(next);
      } catch (reason) {
        fail(reason);
      }
    },
    [save, job, fail],
  );

  const cancel = useCallback(async () => {
    if (!job) return;
    try {
      await apiJson(`/api/qwen/story/jobs/${job.id}/cancel`, { method: 'POST' });
    } catch (reason) {
      fail(reason);
    }
  }, [job, fail]);

  const jobId = job?.id;
  const running = job?.status === 'running';
  useEffect(() => {
    if (!jobId || !running) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const next = await apiJson<StoryJob>(`/api/qwen/story/jobs/${jobId}`);
        if (!alive) return;
        setJob(next);
        if (next.status !== 'running') return;
      } catch (reason) {
        if (alive) fail(reason);
        return;
      }
      timer = setTimeout(tick, POLL_MS);
    };
    timer = setTimeout(tick, POLL_MS);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [jobId, running, fail]);

  // ── profiles ───────────────────────────────────────────────────────────
  /** A recorded or uploaded clip goes to the app's data folder; the profile then points at the stored copy. */
  const uploadClip = useCallback(async (file: File): Promise<string> => {
    const res = await apiFetch(`/api/qwen/clips?name=${encodeURIComponent(file.name)}`, {
      method: 'POST',
      body: await file.arrayBuffer(),
      headers: { 'Content-Type': 'application/octet-stream' },
    });
    return ((await res.json()) as { path: string }).path;
  }, []);

  /** The existing transcription route; the text is a suggestion the user can edit. Empty when it is unavailable. */
  const transcribeClip = useCallback(async (file: File): Promise<string> => {
    try {
      const body = new FormData();
      body.set('audio', file);
      body.set('mode', 'reference');
      body.set('refine', 'false');
      return (await apiJson<{ text: string }>('/transcribe', { method: 'POST', body })).text.trim();
    } catch {
      return '';
    }
  }, []);

  const saveProfile = useCallback(
    async (form: ProfileForm, clip: File | null = null): Promise<Profile | null> => {
      try {
        const ready = clip ? { ...form, ref_audio: await uploadClip(clip) } : form;
        const saved = await apiJson<Profile>(ready.id ? `/api/qwen/profiles/${ready.id}` : '/api/qwen/profiles', {
          method: ready.id ? 'PUT' : 'POST',
          body: JSON.stringify(profileBody(ready)),
        });
        await loadProfiles();
        return saved;
      } catch (reason) {
        fail(reason);
        return null;
      }
    },
    [fail, loadProfiles, uploadClip],
  );

  const prepareProfile = useCallback(
    async (id: string) => {
      try {
        await apiJson(`/api/qwen/profiles/${id}/prepare`, { method: 'POST' });
        await loadProfiles();
      } catch (reason) {
        fail(reason);
      }
    },
    [fail, loadProfiles],
  );

  // A voice being prepared in the background: look again until it is ready or has failed.
  const anyPreparing = profiles.some((p) => p.embedding_state === 'preparing');
  useEffect(() => {
    if (!anyPreparing) return;
    const timer = setInterval(() => void loadProfiles(), 2000);
    return () => clearInterval(timer);
  }, [anyPreparing, loadProfiles]);

  const deleteProfile = useCallback(
    async (id: string) => {
      try {
        await apiJson(`/api/qwen/profiles/${id}`, { method: 'DELETE' });
        await loadProfiles();
      } catch (reason) {
        fail(reason);
      }
    },
    [fail, loadProfiles],
  );

  const importEmbedding = useCallback(
    async (id: string, file: File) => {
      try {
        await apiJson(`/api/qwen/profiles/${id}/embedding`, {
          method: 'PUT',
          body: await file.arrayBuffer(),
          headers: { 'Content-Type': 'application/octet-stream' },
        });
        await loadProfiles();
      } catch (reason) {
        fail(reason);
      }
    },
    [fail, loadProfiles],
  );

  const exportEmbedding = useCallback(
    async (id: string) => {
      try {
        const res = await apiFetch(`/api/qwen/profiles/${id}/embedding`);
        const url = URL.createObjectURL(await res.blob());
        const link = document.createElement('a');
        link.href = url;
        link.download = `${id}.bin`;
        link.click();
        URL.revokeObjectURL(url);
      } catch (reason) {
        fail(reason);
      }
    },
    [fail],
  );

  return {
    state, catalog, profiles, stories, error, setError, storyId, title, script, chunks, narrator, cast, emotions, settings,
    job, busy, running, draft,
    setScript, setTitle: (v: string) => edit(setTitle, v), setChunks: (v: StoryChunk[]) => edit(setChunks, v),
    setNarrator: (v: string | null) => edit(setNarrator, v),
    setCast: (v: Record<string, string>) => edit(setCast, v),
    setEmotions: (v: Record<string, Emotion>) => edit(setEmotions, v),
    setSettings: (v: StorySettings) => edit(setSettings, v),
    applyDraft: (d: Draft) => {
      edit(setChunks, d.chunks);
      setCast(d.cast);
      setEmotions(d.emotions);
    },
    parse, save, open, remove, render, cancel, saveProfile, prepareProfile, transcribeClip, deleteProfile, importEmbedding, exportEmbedding,
  };
}
