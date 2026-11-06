import { useEffect, useSyncExternalStore } from 'react';
import { apiJson } from '@/lib/api/client';

/** A voice effect as the backend lists it (engines/qwen3_studio/effects.py): the built-in ones and the user's own. */
export interface EffectDef {
  id: string;
  label: string;
  builtin: boolean;
  kind: 'chain' | 'graph';
  graph: string;
  aliases: string[];
  codec: 'pcm_s16le' | 'pcm_s24le' | 'pcm_f32le';
}

const FALLBACK_IDS = ['none', 'inner_monologue', 'slight_echo', 'ethereal', 'ghost', 'demon', 'cave', 'radio', 'muffled'];
const FALLBACK: EffectDef[] = FALLBACK_IDS.map((id) => ({
  id,
  label: id === 'none' ? 'None' : id.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase()),
  builtin: true,
  kind: 'chain',
  graph: '',
  aliases: [],
  codec: 'pcm_s16le',
}));

let current: EffectDef[] = FALLBACK;
const listeners = new Set<() => void>();
let requested = false;

function publish(next: EffectDef[]) {
  current = next;
  listeners.forEach((l) => l());
}

/** Reload the list (after adding, editing or deleting an effect). A failed load keeps what is shown. */
export async function reloadEffects(): Promise<void> {
  try {
    const data = await apiJson<{ effects?: EffectDef[] }>('/api/qwen/effects');
    if (Array.isArray(data?.effects) && data.effects.length > 0) publish(data.effects);
  } catch {
    /* the built-in names stay available */
  }
}

/** The effects for a picker. Built-in names are available at once; your own appear when the list has loaded. */
export function useEffects(): EffectDef[] {
  const list = useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => current,
  );
  useEffect(() => {
    if (!requested) {
      requested = true;
      void reloadEffects();
    }
  }, []);
  return list;
}

/** Built-in effects have a translation; yours show the name you gave them. */
export function effectLabel(t: (key: string, options?: Record<string, unknown>) => string, effect: EffectDef): string {
  return effect.builtin ? t(`qwenStory.effect_${effect.id}`, { defaultValue: effect.label }) : effect.label;
}

export function resetEffectsForTest() {
  requested = false;
  publish(FALLBACK);
}
