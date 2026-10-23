import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FolderOpenIcon, LoaderCircleIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { getBridge } from '@/components/bridge';
import type { ExportOptions, ExportResult } from './use-fish-dataset';

const FORMATS = ['jsonl', 'fish_lab', 'ljspeech', 'omnivoice'] as const;
const RATES = [24000, 22050, 16000] as const;
const selectClass =
  'h-8 w-full rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

export function ExportDialog({
  open,
  onOpenChange,
  clips,
  seconds,
  excluded,
  knownTags,
  omnivoiceTags = [],
  onExport,
  onError,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  clips: number;
  seconds: number;
  excluded: number;
  knownTags: Array<[string, number]>;
  omnivoiceTags?: string[];
  onExport: (options: ExportOptions) => Promise<ExportResult | null>;
  onError: (message: string) => void;
}) {
  const { t } = useTranslation();
  const [dest, setDest] = useState('');
  const [name, setName] = useState('fish-dataset');
  const [formats, setFormats] = useState<string[]>(['jsonl', 'fish_lab']);
  const [textMode, setTextMode] = useState<'keep' | 'strip'>('keep');
  const [rate, setRate] = useState('');
  const [val, setVal] = useState('5');
  const [keep, setKeep] = useState('');
  const [level, setLevel] = useState(false);
  const [levelDb, setLevelDb] = useState('-24');
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<ExportResult | null>(null);

  const toggle = (format: string) =>
    setFormats((current) =>
      current.includes(format) ? current.filter((f) => f !== format) : [...current, format],
    );

  const omni = formats.includes('omnivoice');
  const stripped = omni || textMode === 'strip';
  const keepList = keep
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);
  const sourceOf = (entry: string) => entry.split('=')[0].replace(/[[\]]/g, '').trim().toLowerCase();
  const kept = new Set(keepList.map(sourceOf));
  const toggleKeep = (tag: string) =>
    setKeep(
      kept.has(tag)
        ? keepList.filter((entry) => sourceOf(entry) !== tag).join(', ')
        : [...keepList, tag].join(', '),
    );

  const run = async () => {
    setBusy(true);
    setDone(null);
    const result = await onExport({
      dest: dest.trim(),
      name: name.trim() || 'fish-dataset',
      formats,
      text_mode: stripped ? 'strip' : 'keep',
      sample_rate: omni ? 24000 : rate ? Number(rate) : null,
      val_fraction: Math.max(0, Math.min(0.49, Number(val) / 100 || 0)),
      keep_tags: stripped ? keepList : [],
      loudness_db: level && Number.isFinite(Number(levelDb)) ? Number(levelDb) : null,
    });
    setBusy(false);
    if (result) setDone(result);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t('fishDataset.export_title')}</DialogTitle>
          <DialogDescription>
            {t('fishDataset.export_desc', {
              clips,
              minutes: Math.round(seconds / 60),
              excluded,
            })}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 text-sm">
          <div className="space-y-1">
            <label htmlFor="ds-dest" className="text-xs font-medium">
              {t('fishDataset.export_dest')}
            </label>
            <Input
              id="ds-dest"
              className="font-mono text-xs"
              placeholder="D:\\datasets"
              value={dest}
              onChange={(e) => setDest(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="ds-name" className="text-xs font-medium">
              {t('fishDataset.export_name')}
            </label>
            <Input id="ds-name" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} />
            <p className="text-[11px] text-muted-foreground">{t('fishDataset.export_name_hint')}</p>
          </div>

          <fieldset className="space-y-1">
            <legend className="text-xs font-medium">{t('fishDataset.export_formats')}</legend>
            {FORMATS.map((format) => (
              <label key={format} className="flex items-start gap-2 text-xs">
                <input
                  type="checkbox"
                  className="mt-0.5"
                  checked={formats.includes(format)}
                  onChange={() => toggle(format)}
                />
                <span>
                  <b>{t(`fishDataset.format_${format}`)}</b>
                  <span className="block text-muted-foreground">{t(`fishDataset.format_${format}_hint`)}</span>
                </span>
              </label>
            ))}
          </fieldset>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label htmlFor="ds-text" className="text-xs font-medium">
                {t('fishDataset.export_text')}
              </label>
              <select
                id="ds-text"
                className={selectClass}
                value={stripped ? 'strip' : textMode}
                disabled={omni}
                onChange={(e) => setTextMode(e.target.value as 'keep' | 'strip')}
              >
                <option value="keep">{t('fishDataset.text_keep')}</option>
                <option value="strip">{t('fishDataset.text_strip')}</option>
              </select>
            </div>
            <div className="space-y-1">
              <label htmlFor="ds-rate" className="text-xs font-medium">
                {t('fishDataset.export_rate')}
              </label>
              <select
                id="ds-rate"
                className={selectClass}
                value={omni ? '24000' : rate}
                disabled={omni}
                onChange={(e) => setRate(e.target.value)}
              >
                <option value="">{t('fishDataset.rate_asis')}</option>
                {RATES.map((r) => (
                  <option key={r} value={r}>
                    {r} Hz
                  </option>
                ))}
              </select>
            </div>
          </div>
          {omni && (
            <p className="rounded-md bg-muted/40 p-2 text-[11px] leading-snug text-muted-foreground">
              {t('fishDataset.export_omni_note', { tags: omnivoiceTags.join(', ') })}
            </p>
          )}
          {stripped && (
            <div className="space-y-1">
              <label htmlFor="ds-keep" className="text-xs font-medium">
                {t('fishDataset.keep_tags')}
              </label>
              <Input
                id="ds-keep"
                className="font-mono text-xs"
                placeholder="laughing=laughter, sighing=sigh"
                value={keep}
                onChange={(e) => setKeep(e.target.value)}
              />
              <p className="text-[11px] text-muted-foreground">{t('fishDataset.keep_tags_hint')}</p>
              {knownTags.length > 0 && (
                <div className="flex max-h-24 flex-wrap gap-1 overflow-y-auto" aria-label={t('fishDataset.known_tags')}>
                  {knownTags.map(([tag, count]) => (
                    <button
                      key={tag}
                      type="button"
                      aria-pressed={kept.has(tag)}
                      onClick={() => toggleKeep(tag)}
                      className={`rounded-full px-2 py-0.5 text-[10px] ${
                        kept.has(tag) ? 'bg-primary/20 text-primary' : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                      }`}
                    >
                      [{tag}] {count}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          <div className="space-y-1">
            <label className="flex items-center gap-2 text-xs font-medium">
              <input type="checkbox" checked={level} onChange={(e) => setLevel(e.target.checked)} />
              {t('fishDataset.export_loudness')}
            </label>
            {level && (
              <Input
                id="ds-level"
                aria-label={t('fishDataset.export_loudness_db')}
                type="number"
                min={-60}
                max={-3}
                className="w-24"
                value={levelDb}
                onChange={(e) => setLevelDb(e.target.value)}
              />
            )}
            <p className="text-[11px] text-muted-foreground">{t('fishDataset.export_loudness_hint')}</p>
          </div>
          <div className="space-y-1">
            <label htmlFor="ds-val" className="text-xs font-medium">
              {t('fishDataset.export_val')}
            </label>
            <Input
              id="ds-val"
              type="number"
              min={0}
              max={49}
              className="w-24"
              value={val}
              onChange={(e) => setVal(e.target.value)}
            />
            <p className="text-[11px] text-muted-foreground">{t('fishDataset.export_val_hint')}</p>
          </div>

          <p className="rounded-md bg-amber-500/10 p-2 text-[11px] leading-snug text-amber-500">
            {t('fishDataset.terms_note')}
          </p>

          {done && (
            <div role="status" className="space-y-1 rounded-md bg-emerald-500/10 p-2 text-xs text-emerald-500">
              <p>{t('fishDataset.export_done', { clips: done.clips, hours: done.hours })}</p>
              <p className="break-all font-mono text-[11px]">{done.path}</p>
              <Button
                size="xs"
                variant="outline"
                onClick={() =>
                  void getBridge()
                    ?.files.revealPath(done.path)
                    .catch((reason) => onError(String(reason?.message ?? reason)))
                }
              >
                <FolderOpenIcon /> {t('fishStory.reveal')}
              </Button>
            </div>
          )}
        </div>

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('fishStory.done')}</DialogClose>
          <Button disabled={busy || !dest.trim() || formats.length === 0 || clips === 0} onClick={() => void run()}>
            {busy && <LoaderCircleIcon className="animate-spin" />} {t('fishDataset.export_button')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
