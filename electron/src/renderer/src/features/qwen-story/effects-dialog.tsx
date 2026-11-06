import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { apiJson, describeError } from '@/lib/api/client';
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

const BLANK: Form = { label: '', aliases: '', kind: 'chain', graph: '', codec: 'pcm_s16le' };

const toForm = (e: EffectDef): Form => ({ id: e.id, label: e.label, aliases: e.aliases.join(', '), kind: e.kind, graph: e.graph, codec: e.codec });
const toBody = (f: Form) => ({
  id: f.id ?? null,
  label: f.label.trim(),
  kind: f.kind,
  graph: f.graph.trim(),
  codec: f.codec,
  aliases: f.aliases.split(',').map((a) => a.trim()).filter(Boolean),
});

/** Add, edit, test and delete your own ffmpeg voice effects. The built-in ones can be copied (to start from one) but not changed. */
export function EffectsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t } = useTranslation();
  const effects = useEffects();
  const [form, setForm] = useState<Form | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const set = (patch: Partial<Form>) => {
    setForm((f) => (f ? { ...f, ...patch } : f));
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
  const save = () =>
    run(async () => {
      const f = form as Form;
      await apiJson(f.id ? `/api/qwen/effects/${encodeURIComponent(f.id)}` : '/api/qwen/effects', {
        method: f.id ? 'PUT' : 'POST',
        body: JSON.stringify(toBody(f)),
      });
      await reloadEffects();
      setForm(null);
    });
  const remove = (id: string) =>
    run(async () => {
      await apiJson(`/api/qwen/effects/${encodeURIComponent(id)}`, { method: 'DELETE' });
      await reloadEffects();
      setForm(null);
    });

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
                <Button size="xs" variant="outline" onClick={() => setForm(e.builtin ? { ...toForm(e), id: undefined, label: `${effectLabel(t, e)} copy`, aliases: '' } : toForm(e))}>
                  {e.builtin ? t('qwenStory.effects_copy') : t('qwenStory.effects_edit')}
                </Button>
              </span>
            </li>
          ))}
        </ul>
        {!form && (
          <Button size="sm" variant="outline" onClick={() => setForm({ ...BLANK })}>
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
              {form.id && (
                <Button size="sm" variant="destructive" disabled={busy} onClick={() => void remove(form.id as string)}>
                  {t('qwenStory.effects_delete')}
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={() => setForm(null)}>
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
