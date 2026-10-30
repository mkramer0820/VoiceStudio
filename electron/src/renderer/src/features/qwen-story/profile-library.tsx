import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDownIcon, DownloadIcon, LoaderCircleIcon, PencilIcon, PlusIcon, RefreshCwIcon, Trash2Icon, UploadIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { ReferenceSourcePicker } from '@/features/clone/reference-input';
import type { QwenModel } from '../qwen-playground/qwen-model';
import { EmotionPicker, selectClass } from './emotion-picker';
import {
  blankForm,
  formFromProfile,
  modelsForKind,
  profileProblem,
  type EmotionCatalog,
  type Profile,
  type ProfileForm,
  type ProfileKind,
} from './story-model';

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block space-y-1">
      <span className="block text-xs text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}

function ProfileDialog({
  form,
  setForm,
  models,
  catalog,
  onSave,
  onTranscribe,
}: {
  form: ProfileForm | null;
  setForm: (f: ProfileForm | null) => void;
  models: QwenModel[];
  catalog: EmotionCatalog | null;
  onSave: (f: ProfileForm, clip: File | null) => Promise<boolean>;
  onTranscribe: (file: File) => Promise<string>;
}) {
  const { t } = useTranslation();
  const [clip, setClip] = useState<File | null>(null);
  const [clipUrl, setClipUrl] = useState('');
  const [transcribing, setTranscribing] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const formRef = useRef(form);
  formRef.current = form;
  useEffect(() => {
    if (!clip) {
      setClipUrl('');
      return;
    }
    const url = URL.createObjectURL(clip);
    setClipUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [clip]);
  // A new clip is transcribed for the user; the text stays editable, and a typed text is never overwritten.
  const acceptClip = (file: File) => {
    setClip(file);
    setTranscribing(true);
    void onTranscribe(file).then((text) => {
      setTranscribing(false);
      const cur = formRef.current;
      if (text && cur && !cur.ref_text.trim()) setForm({ ...cur, ref_text: text });
    });
  };
  if (!form) return null;
  const set = (patch: Partial<ProfileForm>) => setForm({ ...form, ...patch });
  const usable = modelsForKind(form.kind, models);
  const model = models.find((m) => m.id === form.model_id);
  const problem = profileProblem(form, clip !== null);
  const retarget = (kind: ProfileKind) => {
    const pool = modelsForKind(kind, models);
    const next = pool.find((m) => m.id === form.model_id) ?? pool[0];
    set({ kind, model_id: next?.id ?? '', speaker: next?.speakers.length === 1 ? next.speakers[0] : '' });
    if (kind !== 'clone') setClip(null);
  };
  return (
    <Dialog open onOpenChange={(open) => !open && setForm(null)}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{form.id ? t('qwenStory.profile_edit') : t('qwenStory.profile_new')}</DialogTitle>
          <DialogDescription>{t('qwenStory.profile_desc')}</DialogDescription>
        </DialogHeader>
        <div className="space-y-3 text-sm">
          <Field label={t('qwenStory.profile_name')}>
            <Input value={form.name} onChange={(e) => set({ name: e.target.value })} />
          </Field>
          <Field label={t('qwenStory.profile_kind')}>
            <select className={`${selectClass} w-full`} value={form.kind} onChange={(e) => retarget(e.target.value as ProfileKind)}>
              <option value="narrator">{t('qwenStory.kind_narrator')}</option>
              <option value="stock">{t('qwenStory.kind_stock')}</option>
              <option value="clone">{t('qwenStory.kind_clone')}</option>
            </select>
          </Field>
          <Field label={t('qwenStory.profile_model')}>
            <select className={`${selectClass} w-full`} value={form.model_id} onChange={(e) => set({ model_id: e.target.value })}>
              {usable.length === 0 && <option value="">{t('qwenStory.no_model_for_kind')}</option>}
              {usable.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
            <span className="block text-xs text-muted-foreground">{t('qwenStory.model_hint')}</span>
          </Field>
          {form.kind !== 'clone' ? (
            <Field label={t('qwenStory.profile_speaker')}>
              <select className={`${selectClass} w-full`} value={form.speaker} onChange={(e) => set({ speaker: e.target.value })}>
                <option value="">{t('qwenStory.speaker_choose')}</option>
                {(model?.speakers ?? []).map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
                {(model?.borrowable_speakers ?? []).map((s) => (
                  <option key={s} value={s}>
                    {s} ({t('qwenStory.borrowed')})
                  </option>
                ))}
              </select>
            </Field>
          ) : (
            <>
              <div className="space-y-1">
                <span className="block text-xs text-muted-foreground">{t('qwenStory.voice_sample')}</span>
                <ReferenceSourcePicker onAccept={acceptClip} />
                <span className="block text-xs text-muted-foreground">{t('qwenStory.voice_sample_hint')}</span>
                {clip && clipUrl && (
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <audio controls src={clipUrl} aria-label={clip.name} className="h-8 w-56 max-w-full" />
                    <span>{clip.name}</span>
                  </div>
                )}
                {!clip && form.id && form.ref_audio && <span className="block text-xs text-emerald-500">{t('qwenStory.sample_kept')}</span>}
              </div>
              <Field label={t('qwenStory.voice_words')}>
                <textarea
                  rows={2}
                  className="w-full resize-y rounded-md border border-input bg-input/20 p-2 text-sm"
                  value={form.ref_text}
                  placeholder={transcribing ? t('qwenStory.transcribing') : ''}
                  onChange={(e) => set({ ref_text: e.target.value })}
                />
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  {transcribing && <LoaderCircleIcon className="size-3 animate-spin" />}
                  {t('qwenStory.voice_words_hint')}
                </span>
              </Field>
            </>
          )}
          <Field label={t('qwenStory.profile_instruct')}>
            <Input value={form.default_instruct} onChange={(e) => set({ default_instruct: e.target.value })} />
          </Field>
          {form.kind === 'clone' && (
            <div>
              <button
                type="button"
                aria-expanded={advanced}
                className="flex items-center gap-1 text-xs text-muted-foreground"
                onClick={() => setAdvanced(!advanced)}
              >
                <ChevronDownIcon className={`size-3.5 transition-transform ${advanced ? '' : '-rotate-90'}`} /> {t('qwenStory.advanced')}
              </button>
              {advanced && (
                <label className="mt-1 flex items-center gap-2 text-xs">
                  <input type="checkbox" checked={form.xvec_only} onChange={(e) => set({ xvec_only: e.target.checked })} />
                  {t('qwenStory.profile_xvec_only')}
                </label>
              )}
            </div>
          )}
          <div className="space-y-1">
            <span className="block text-xs text-muted-foreground">{t('qwenStory.profile_emotion')}</span>
            <EmotionPicker catalog={catalog} value={form.emotion} label={t('qwenStory.profile_emotion')} onChange={(v) => set({ emotion: v ?? form.emotion })} />
            {form.kind === 'narrator' && <p className="text-xs text-amber-500">{t('qwenStory.unverified_narrator')}</p>}
          </div>
          <div className="grid grid-cols-2 gap-2">
            {(['temperature', 'top_k', 'top_p', 'repetition_penalty'] as const).map((key) => (
              <Field key={key} label={t(`qwenStory.sampling_${key}`)}>
                <Input inputMode="decimal" value={form[key]} placeholder={t('qwenStory.engine_default')} onChange={(e) => set({ [key]: e.target.value })} />
              </Field>
            ))}
          </div>
          <Field label={t('qwenStory.profile_notes')}>
            <textarea rows={2} className="w-full resize-y rounded-md border border-input bg-input/20 p-2 text-sm" value={form.notes} onChange={(e) => set({ notes: e.target.value })} />
          </Field>
          {problem && (
            <p role="alert" className="text-xs text-amber-500">
              {t(problem)}
            </p>
          )}
        </div>
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('qwenStory.cancel')}</DialogClose>
          <Button disabled={!!problem || transcribing} onClick={() => void onSave(form, clip).then((ok) => ok && setForm(null))}>
            {form.kind === 'clone' ? t('qwenStory.save_voice') : t('qwenStory.profile_save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** The profile library: create, edit, delete, import and export the cached speaker embedding. */
export function ProfileLibrary({
  profiles,
  models,
  catalog,
  onSave,
  onTranscribe,
  onPrepare,
  onDelete,
  onImport,
  onExport,
}: {
  profiles: Profile[];
  models: QwenModel[];
  catalog: EmotionCatalog | null;
  onSave: (f: ProfileForm, clip: File | null) => Promise<boolean>;
  onTranscribe: (file: File) => Promise<string>;
  onPrepare: (id: string) => void;
  onDelete: (id: string) => void;
  onImport: (id: string, file: File) => void;
  onExport: (id: string) => void;
}) {
  const { t } = useTranslation();
  const [form, setForm] = useState<ProfileForm | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const importFor = useRef<string | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  return (
    <section aria-label={t('qwenStory.profiles_title')} className="space-y-2 rounded-xl border border-border/60 bg-card/30 p-3">
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-medium">{t('qwenStory.profiles_title')}</h2>
        <Button size="sm" variant="outline" className="ml-auto" onClick={() => setForm(blankForm(models))}>
          <PlusIcon /> {t('qwenStory.profile_new')}
        </Button>
      </div>
      {profiles.length === 0 && <p className="text-xs text-muted-foreground">{t('qwenStory.profiles_empty')}</p>}
      <ul className="space-y-1">
        {profiles.map((p) => (
          <li key={p.id} className="flex flex-wrap items-center gap-2 rounded-md bg-background/60 p-2 text-xs">
            <span className="font-medium">{p.name}</span>
            <span className="text-muted-foreground">{t(`qwenStory.kind_${p.kind}`)}</span>
            <span className="text-muted-foreground">
              {p.kind === 'clone' ? (p.xvec_only ? t('qwenStory.xvec_only_short') : t('qwenStory.clip_and_text')) : p.speaker}
            </span>
            {!models.some((m) => m.id === p.model_id) && <span className="text-amber-500">{t('qwenStory.model_missing_short')}</span>}
            {p.kind === 'clone' && p.embedding_state === 'preparing' && (
              <span className="flex items-center gap-1 text-muted-foreground" role="status">
                <LoaderCircleIcon className="size-3 animate-spin" /> {t('qwenStory.voice_preparing')}
              </span>
            )}
            {p.kind === 'clone' && p.embedding_state === 'ready' && <span className="text-emerald-500">{t('qwenStory.voice_ready')}</span>}
            {p.kind === 'clone' && p.embedding_state === 'failed' && (
              <span className="flex items-center gap-1 text-amber-500" title={p.embedding_error}>
                {t('qwenStory.voice_prepare_failed')}
                <Button size="icon-xs" variant="ghost" aria-label={t('qwenStory.voice_retry')} onClick={() => onPrepare(p.id)}>
                  <RefreshCwIcon />
                </Button>
              </span>
            )}
            {p.kind === 'clone' && p.embedding_state === 'none' && <span className="text-muted-foreground">{t('qwenStory.voice_prepare_later')}</span>}
            <span className="ml-auto flex gap-1">
              <Button size="icon-xs" variant="ghost" aria-label={t('qwenStory.profile_edit')} onClick={() => setForm(formFromProfile(p))}>
                <PencilIcon />
              </Button>
              <Button size="icon-xs" variant="ghost" aria-label={t('qwenStory.profile_delete')} onClick={() => onDelete(p.id)}>
                <Trash2Icon />
              </Button>
            </span>
          </li>
        ))}
      </ul>
      <div>
        <button type="button" aria-expanded={advanced} className="flex items-center gap-1 text-xs text-muted-foreground" onClick={() => setAdvanced(!advanced)}>
          <ChevronDownIcon className={`size-3.5 transition-transform ${advanced ? '' : '-rotate-90'}`} /> {t('qwenStory.advanced')}
        </button>
        {advanced && (
          <div className="mt-1 space-y-1 text-xs">
            <p className="text-muted-foreground">{t('qwenStory.embedding_hint')}</p>
            {profiles.filter((p) => p.kind === 'clone').map((p) => (
              <div key={p.id} className="flex items-center gap-2">
                <span className="w-40 truncate">{p.name}</span>
                <Button
                  size="xs"
                  variant="outline"
                  aria-label={t('qwenStory.embedding_import_for', { name: p.name })}
                  onClick={() => {
                    importFor.current = p.id;
                    picker.current?.click();
                  }}
                >
                  <UploadIcon /> {t('qwenStory.embedding_import')}
                </Button>
                <Button size="xs" variant="outline" disabled={!p.has_embedding} aria-label={t('qwenStory.embedding_export_for', { name: p.name })} onClick={() => onExport(p.id)}>
                  <DownloadIcon /> {t('qwenStory.embedding_export')}
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
      <input
        ref={picker}
        type="file"
        accept=".bin"
        hidden
        data-testid="embedding-file"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file && importFor.current) onImport(importFor.current, file);
          e.target.value = '';
        }}
      />
      <ProfileDialog form={form} setForm={setForm} models={models} catalog={catalog} onSave={onSave} onTranscribe={onTranscribe} />
    </section>
  );
}
