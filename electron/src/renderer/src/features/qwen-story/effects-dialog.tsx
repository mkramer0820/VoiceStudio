import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { apiFetch, apiJson, describeError } from '@/lib/api/client';
import { selectClass } from './emotion-picker';
import { effectLabel, reloadEffects, useEffects, type EffectDef } from './effects-model';

interface Form {
  id?: string;
  label: string;
  aliases: string;
  kind: 'chain' | 'graph';
  graph: string;
  codec: EffectDef['codec'];
}

interface Sample {
  blob: Blob;
  name: string;
}

const BLANK: Form = { label: '', aliases: '', kind: 'chain', graph: '', codec: 'pcm_s16le' };

const toForm = (e: EffectDef, label: string): Form => ({ id: e.id, label, aliases: e.aliases.join(', '), kind: e.kind, graph: e.graph, codec: e.codec });
const toBody = (f: Form) => ({
  id: f.id ?? null,
  label: f.label.trim(),
  kind: f.kind,
  graph: f.graph.trim(),
  codec: f.codec,
  aliases: f.aliases.split(',').map((a) => a.trim()).filter(Boolean),
});

/**
 * Add, edit, listen to and delete voice effects. Built-in effects can be edited too (your version is stored under the same id) and reset to
 * the shipped recipe. **Hear it** runs the recipe on a clip you choose and plays the original and the processed audio one after the other, so
 * a recipe is judged by ear, not just by "ffmpeg accepted it". `renderSamples` lets a page offer its own clips (for example rendered lines).
 */
export function EffectsDialog({
  open,
  onOpenChange,
  renderSamples,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  renderSamples?: (pick: (blob: Blob, name: string) => void) => ReactNode;
}) {
  const { t } = useTranslation();
  const effects = useEffects();
  const [form, setForm] = useState<Form | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  const [urls, setUrls] = useState<{ original: string; processed: string } | null>(null);
  const live = useRef<string[]>([]);

  const forget = () => {
    live.current.forEach((u) => URL.revokeObjectURL(u));
    live.current = [];
    setUrls(null);
  };
  useEffect(() => forget, []);
  useEffect(() => {
    if (!open) forget();
  }, [open]);

  const set = (patch: Partial<Form>) => {
    setForm((f) => (f ? { ...f, ...patch } : f));
    setMessage(null);
  };
  const pick = (blob: Blob, name: string) => {
    forget();
    setSample({ blob, name });
    setMessage(null);
  };

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    try {
      await action();
    } catch (cause) {
      setMessage({ ok: false, text: describeError(cause) });
    } finally {
      setBusy(false);
    }
  };
  const test = () =>
    run(async () => {
      const r = await apiJson<{ ok: boolean; channels: number | null; error: string | null }>('/api/qwen/effects/test', {
        method: 'POST',
        body: JSON.stringify(toBody(form as Form)),
      });
      setMessage(r.ok ? { ok: true, text: t('qwenStory.effects_test_ok', { channels: r.channels ?? 1 }) } : { ok: false, text: r.error ?? '' });
    });
  const hear = () =>
    run(async () => {
      if (!sample) return;
      const body = new FormData();
      body.append('file', sample.blob, sample.name);
      body.append('definition', JSON.stringify(toBody(form as Form)));
      const res = await apiFetch('/api/qwen/effects/preview', { method: 'POST', body });
      const processed = URL.createObjectURL(await res.blob());
      const original = URL.createObjectURL(sample.blob);
      forget();
      live.current = [original, processed];
      setUrls({ original, processed });
      setMessage(null);
    });
  const save = () =>
    run(async () => {
      const f = form as Form;
      await apiJson(f.id ? `/api/qwen/effects/${encodeURIComponent(f.id)}` : '/api/qwen/effects', {
        method: f.id ? 'PUT' : 'POST',
        body: JSON.stringify(toBody(f)),
      });
      await reloadEffects();
      setForm(null);
      forget();
    });
  const remove = (id: string) =>
    run(async () => {
      await apiJson(`/api/qwen/effects/${encodeURIComponent(id)}`, { method: 'DELETE' });
      await reloadEffects();
      setForm(null);
      forget();
    });

  const editing = form?.id ? effects.find((e) => e.id === form.id) : undefined;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t('qwenStory.effects_title')}</DialogTitle>
          <DialogDescription>{t('qwenStory.effects_desc')}</DialogDescription>
        </DialogHeader>
        <ul className="space-y-1 text-sm">
          {effects.filter((e) => e.id !== 'none').map((e) => (
            <li key={e.id} className="flex items-center gap-2">
              <span className="w-40 truncate font-medium">{effectLabel(t, e)}</span>
              <code className="truncate text-xs text-muted-foreground">{e.aliases.length ? `{${e.aliases[0]}}` : `{${e.id}}`}</code>
              <span className="ml-auto flex gap-1">
                <Button size="xs" variant="outline" onClick={() => { setForm(toForm(e, effectLabel(t, e))); forget(); }}>
                  {t('qwenStory.effects_edit')}
                </Button>
                {e.builtin && (
                  <Button size="xs" variant="ghost" onClick={() => { setForm({ ...toForm(e, `${effectLabel(t, e)} copy`), id: undefined, aliases: '' }); forget(); }}>
                    {t('qwenStory.effects_copy')}
                  </Button>
                )}
              </span>
            </li>
          ))}
        </ul>
        {!form && (
          <Button size="sm" variant="outline" onClick={() => { setForm({ ...BLANK }); forget(); }}>
            {t('qwenStory.effects_new')}
          </Button>
        )}
        {form && (
          <div className="space-y-2 rounded-lg border border-border/60 p-3 text-sm">
            <label className="block text-xs font-medium">
              {t('qwenStory.effects_name')}
              <Input value={form.label} onChange={(e) => set({ label: e.target.value })} />
            </label>
            <label className="block text-xs font-medium">
              {t('qwenStory.effects_aliases')}
              <Input value={form.aliases} placeholder="internal, thought" onChange={(e) => set({ aliases: e.target.value })} />
            </label>
            <div className="flex gap-2">
              <label className="block text-xs font-medium">
                {t('qwenStory.effects_type')}
                <select className={`${selectClass} mt-1 block w-48`} value={form.kind} onChange={(e) => set({ kind: e.target.value as Form['kind'] })}>
                  <option value="chain">{t('qwenStory.effects_type_chain')}</option>
                  <option value="graph">{t('qwenStory.effects_type_graph')}</option>
                </select>
              </label>
              <label className="block text-xs font-medium">
                {t('qwenStory.effects_codec')}
                <select className={`${selectClass} mt-1 block w-36`} value={form.codec} onChange={(e) => set({ codec: e.target.value as Form['codec'] })}>
                  <option value="pcm_s16le">16-bit</option>
                  <option value="pcm_s24le">24-bit</option>
                  <option value="pcm_f32le">32-bit float</option>
                </select>
              </label>
            </div>
            <label className="block text-xs font-medium">
              {t('qwenStory.effects_graph')}
              <textarea
                rows={5}
                spellCheck={false}
                className="mt-1 w-full resize-y rounded-md border border-input bg-input/20 p-2 font-mono text-xs"
                value={form.graph}
                onChange={(e) => set({ graph: e.target.value })}
              />
            </label>
            <p className="text-xs text-muted-foreground">{form.kind === 'graph' ? t('qwenStory.effects_graph_hint') : t('qwenStory.effects_chain_hint')}</p>

            <section className="space-y-2 rounded-md border border-border/50 p-2">
              <h4 className="text-xs font-medium">{t('qwenStory.effects_hear_title')}</h4>
              <div className="flex flex-wrap items-center gap-2">
                <input
                  type="file"
                  accept="audio/*,video/*"
                  aria-label={t('qwenStory.effects_sample')}
                  className="text-xs"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) pick(file, file.name);
                  }}
                />
                {renderSamples?.(pick)}
              </div>
              {sample && <p className="text-xs text-muted-foreground">{t('qwenStory.effects_sample_chosen', { name: sample.name })}</p>}
              <Button size="sm" disabled={busy || !sample || !form.graph.trim()} onClick={() => void hear()}>
                {t('qwenStory.effects_hear')}
              </Button>
              {urls && (
                <div className="grid gap-2 sm:grid-cols-2">
                  <label className="block text-xs">
                    {t('qwenStory.effects_original')}
                    <audio controls src={urls.original} className="mt-1 w-full" />
                  </label>
                  <label className="block text-xs">
                    {t('qwenStory.effects_processed')}
                    <audio controls src={urls.processed} className="mt-1 w-full" autoPlay />
                  </label>
                </div>
              )}
            </section>

            {message && (
              <p role={message.ok ? 'status' : 'alert'} className={message.ok ? 'text-xs text-emerald-500' : 'text-xs text-destructive'}>
                {message.text}
              </p>
            )}
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" disabled={busy || !form.graph.trim()} onClick={() => void test()}>
                {t('qwenStory.effects_test')}
              </Button>
              <Button size="sm" disabled={busy || !form.label.trim() || !form.graph.trim()} onClick={() => void save()}>
                {t('common.save')}
              </Button>
              {form.id && editing && (!editing.builtin || editing.modified) && (
                <Button size="sm" variant="destructive" disabled={busy} onClick={() => void remove(form.id as string)}>
                  {editing.builtin ? t('qwenStory.effects_reset') : t('qwenStory.effects_delete')}
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={() => { setForm(null); forget(); }}>
                {t('common.cancel')}
              </Button>
            </div>
          </div>
        )}
        <p className="text-xs text-muted-foreground">{t('qwenStory.effects_tag_hint')}</p>
      </DialogContent>
    </Dialog>
  );
}
