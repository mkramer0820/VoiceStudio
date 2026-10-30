import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { CpuIcon, FolderOpenIcon, LoaderCircleIcon } from 'lucide-react';
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
import {
  DEFAULT_LEVEL_DB,
  EMPTY_QWEN,
  EMPTY_QWEN_TRAIN,
  EMPTY_TRAINING,
  invalidKeys,
  keepTagsOf,
  levelInvalid,
  MAX_KEPT_TAGS,
  QWEN_FORMAT,
  qwenBody,
  qwenInvalid,
  qwenTrainBody,
  qwenTrainInvalid,
  trainingBody,
  unknownTags,
  type QwenForm,
  type QwenTrainForm,
  type TagActions,
  type TrainingForm,
} from './export-options';
import { LoudnessSection } from './export-loudness';
import { QwenSection } from './export-qwen';
import { ReferencePanel, type ReferenceClip } from './export-reference';
import { TagsSection } from './export-tags';
import { QwenTrainingSection, TrainingSection } from './export-training';
import type {
  AnalysisStatus,
  ExportOptions,
  ExportResult,
  LevelSummary,
  LoudnessReport,
  ReferenceSuggestion,
  ShortlistClip,
} from './use-fish-dataset';

const FORMATS = ['jsonl', 'fish_lab', 'ljspeech', 'omnivoice', QWEN_FORMAT] as const;
const DEFAULT_FORMATS = ['jsonl', 'fish_lab'];
const RATES = [24000, 22050, 16000] as const;
const selectClass =
  'h-8 w-full rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

/** What level matching did: the target, how many clips changed, each chapter's median before and after, and the peak-limited ones. */
function LoudnessResult({ report }: { report: LoudnessReport }) {
  const { t } = useTranslation();
  const chapters = Object.entries(report.chapters ?? {});
  return (
    <div className="space-y-0.5" data-testid="loudness-result">
      <p>
        {t('fishDataset.loudness_done', {
          target: report.target_rms_db,
          adjusted: report.adjusted,
          clips: report.clips ?? report.adjusted,
        })}
      </p>
      {chapters.length > 0 && (
        <ul className="max-h-24 overflow-y-auto text-[11px] tabular-nums">
          {chapters.map(([chapter, c]) => (
            <li key={chapter}>
              {t('fishDataset.loudness_chapter', { chapter, before: c.median_before_db, after: c.median_after_db })}
            </li>
          ))}
        </ul>
      )}
      {report.peak_limited > 0 ? (
        <p className="text-[11px] text-amber-500">
          {t('fishDataset.loudness_limited', { count: report.peak_limited, db: report.worst_shortfall_db ?? 0 })}
        </p>
      ) : (
        <p className="text-[11px]">{t('fishDataset.loudness_none_limited')}</p>
      )}
    </div>
  );
}

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
  onTrain,
  reference = null,
  onReferenceChange = () => {},
  shortlist = [],
  onRemoveShortlist,
  onSuggestReferences = async () => ({
    speaker: { voice_id: '', name: '' },
    considered: 0,
    unmeasured: 0,
    clips: 0,
    candidates: [],
  }),
  levels = null,
  analysis = null,
  onMeasure,
  speakerCount = 1,
  speakerId = '',
  speakerName = '',
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
  /** Open the Train step for the export that was just written. */
  onTrain?: (exportDir: string) => void;
  /** Qwen3-TTS: the one reference clip the user chose, or null for an automatic choice. */
  reference?: ReferenceClip | null;
  onReferenceChange?: (clip: ReferenceClip | null) => void;
  /** Clips the user starred as reference candidates. */
  shortlist?: ShortlistClip[];
  onRemoveShortlist?: (keys: string[]) => void;
  /** Ranked reference candidates for the current selection (measured clips only). */
  onSuggestReferences?: (limit: number, minDuration: number, maxDuration: number) => Promise<ReferenceSuggestion>;
  /** Level spread of the selection, to help choose the loudness target. */
  levels?: LevelSummary | null;
  analysis?: AnalysisStatus | null;
  /** Start the "Measure audio" run. */
  onMeasure?: () => void;
  /** The single speaker being exported, when known. */
  speakerId?: string;
  /** How many speakers the current selection spans (Qwen3-TTS trains exactly one). */
  speakerCount?: number;
  /** The selected voice's name, shown in the checkpoint path until a speaker name is typed. */
  speakerName?: string;
}) {
  const { t } = useTranslation();
  const [dest, setDest] = useState('');
  const [name, setName] = useState('fish-dataset');
  const [formats, setFormats] = useState<string[]>(DEFAULT_FORMATS);
  const [textMode, setTextMode] = useState<'keep' | 'strip'>('keep');
  const [rate, setRate] = useState('');
  const [val, setVal] = useState('5');
  const [tagActions, setTagActions] = useState<TagActions>({});
  const [strict, setStrict] = useState(true);
  const [training, setTraining] = useState(true);
  const [trainingForm, setTrainingForm] = useState<TrainingForm>(EMPTY_TRAINING);
  const [qwenForm, setQwenForm] = useState<QwenForm>(EMPTY_QWEN);
  const [qwenTrain, setQwenTrain] = useState<QwenTrainForm>(EMPTY_QWEN_TRAIN);
  // Matching loudness is on by default for the training formats; once the user touches the box it stays as they set it.
  const [levelChoice, setLevelChoice] = useState<boolean | null>(null);
  const [levelDb, setLevelDb] = useState(DEFAULT_LEVEL_DB);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<ExportResult | null>(null);

  // The Qwen3-TTS folder is a complete layout of its own, so it replaces the other formats.
  const toggle = (format: string) =>
    setFormats((current) => {
      if (format === QWEN_FORMAT) return current.includes(QWEN_FORMAT) ? DEFAULT_FORMATS : [QWEN_FORMAT];
      return current.includes(format) ? current.filter((f) => f !== format) : [...current, format];
    });

  const qwen = formats.includes(QWEN_FORMAT);
  const omni = formats.includes('omnivoice');
  const level = levelChoice ?? (qwen || omni);
  const levelBad = level && levelInvalid(levelDb);
  const stripped = omni || qwen || textMode === 'strip';
  const tagNames = knownTags.map(([tag]) => tag);
  const unknown = omni && strict ? unknownTags(tagNames, tagActions, omnivoiceTags) : [];
  const keptEntries = stripped && !qwen ? keepTagsOf(tagNames, tagActions) : [];
  const tooManyTags = keptEntries.length > MAX_KEPT_TAGS;
  const writeTraining = omni && training;
  const trainSeconds = seconds * (1 - Math.max(0, Math.min(0.49, Number(val) / 100 || 0)));
  const trainingInvalid = writeTraining && invalidKeys(trainingForm).length > 0;
  const qwenBlocked = qwen && (qwenInvalid(qwenForm).length > 0 || qwenTrainInvalid(qwenTrain).length > 0 || (qwenForm.multi ? speakerCount > 4 : speakerCount > 1) || levelBad);

  const run = async () => {
    setBusy(true);
    setDone(null);
    const result = await onExport({
      dest: dest.trim(),
      name: name.trim() || 'fish-dataset',
      formats,
      text_mode: stripped ? 'strip' : 'keep',
      sample_rate: omni || qwen ? 24000 : rate ? Number(rate) : null,
      val_fraction: qwen ? 0 : Math.max(0, Math.min(0.49, Number(val) / 100 || 0)),
      keep_tags: keptEntries,
      strict_tags: omni ? strict : undefined,
      training: qwen ? qwenTrainBody(qwenTrain) : writeTraining ? trainingBody(trainingForm) : undefined,
      qwen3: qwen ? qwenBody(qwenForm, reference?.key ?? null) : undefined,
      loudness_db: level && !levelInvalid(levelDb) ? Number(levelDb) : null,
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
                  disabled={qwen && format !== QWEN_FORMAT}
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
                disabled={omni || qwen}
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
                value={omni || qwen ? '24000' : rate}
                disabled={omni || qwen}
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
          <LoudnessSection
            enabled={level}
            onEnabledChange={setLevelChoice}
            value={levelDb}
            onValueChange={setLevelDb}
            levels={levels}
            analysis={analysis}
            onMeasure={onMeasure}
          />
          {omni && (
            <p className="rounded-md bg-muted/40 p-2 text-[11px] leading-snug text-muted-foreground">
              {t('fishDataset.export_omni_note', { tags: omnivoiceTags.join(', ') })}
            </p>
          )}
          {qwen && (
            <QwenSection
              form={qwenForm}
              onChange={setQwenForm}
              speakerCount={speakerCount}
              referencePanel={
                <ReferencePanel
                  open={open}
                  speakerId={speakerId}
                  shortlist={shortlist}
                  reference={reference}
                  onChange={onReferenceChange}
                  onSuggest={onSuggestReferences}
                  minDuration={Number(qwenForm.minDuration) > 0 ? Number(qwenForm.minDuration) : 0.8}
                  maxDuration={Number(qwenForm.maxDuration) > 0 ? Number(qwenForm.maxDuration) : 30}
                  analysis={analysis}
                  onMeasure={onMeasure}
                  onRemoveShortlist={onRemoveShortlist}
                  onError={onError}
                />
              }
            />
          )}
          {stripped && !qwen && (
            <TagsSection
              knownTags={knownTags}
              actions={tagActions}
              onChange={setTagActions}
              unknown={unknown}
              omnivoiceTags={omnivoiceTags}
              checkKnown={omni && strict}
            />
          )}
          {omni && (
            <label className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={strict} onChange={(e) => setStrict(e.target.checked)} />
              {t('fishDataset.strict_tags')}
            </label>
          )}
          {!qwen && (
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
          )}

          {qwen && (
            <details className="rounded-md border border-border/50 p-2" open>
              <summary className="cursor-pointer text-xs font-medium">{t('fishDataset.train_setup')}</summary>
              <div className="mt-2">
                <QwenTrainingSection form={qwenTrain} onChange={setQwenTrain} clips={clips} speaker={qwenForm.speaker || speakerName} />
              </div>
            </details>
          )}
          {omni && (
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-xs font-medium">
                <input type="checkbox" checked={training} onChange={(e) => setTraining(e.target.checked)} />
                {t('fishDataset.train_write')}
              </label>
              {training && (
                <details className="rounded-md border border-border/50 p-2">
                  <summary className="cursor-pointer text-xs font-medium">{t('fishDataset.train_setup')}</summary>
                  <div className="mt-2">
                    <TrainingSection form={trainingForm} onChange={setTrainingForm} seconds={trainSeconds} />
                  </div>
                </details>
              )}
            </div>
          )}

          <p className="rounded-md bg-amber-500/10 p-2 text-[11px] leading-snug text-amber-500">
            {t('fishDataset.terms_note')}
          </p>

          {done && (
            <div role="status" className="space-y-1 rounded-md bg-emerald-500/10 p-2 text-xs text-emerald-500">
              <p>{t('fishDataset.export_done', { clips: done.clips, hours: done.hours })}</p>
              <p className="break-all font-mono text-[11px]">{done.path}</p>
              {done.loudness && <LoudnessResult report={done.loudness} />}
              {done.qwen3 && (
                <>
                  <p>{t('fishDataset.qwen_done', { accepted: done.qwen3.accepted, rejected: done.qwen3.rejected })}</p>
                  <p className="break-all text-[11px]">
                    {t('fishDataset.qwen_ref_used', { source: done.qwen3.reference.source ?? '' })}
                  </p>
                  {done.qwen3.zip && <p className="break-all font-mono text-[11px]">{done.qwen3.zip.path}</p>}
                  <p>{t('fishDataset.train_written', { files: done.qwen3.files.join(', ') })}</p>
                  <p className="font-medium">{t('fishDataset.qtr_start')}</p>
                  <pre className="whitespace-pre-wrap break-all text-[10px]" data-testid="qwen-start-commands">
                    {[...done.qwen3.commands.notebook, '', ...done.qwen3.commands.colab_cli].join('\n')}
                  </pre>
                  <pre className="whitespace-pre-wrap break-all text-[10px]">
                    {[done.qwen3.commands.official.prepare, done.qwen3.commands.official.train].join('\n')}
                  </pre>
                  <p className="break-all text-[11px]">
                    {t('fishDataset.qtr_checkpoints_on_drive', { path: done.qwen3.commands.official.checkpoint_dir })}
                  </p>
                  {done.qwen3.warnings.map((w) => (
                    <p key={w.code} className="text-[11px] text-amber-500">
                      {w.message}
                    </p>
                  ))}
                </>
              )}
              {done.training && (
                <>
                  <p>{t('fishDataset.train_written', { files: done.training.files.join(', ') })}</p>
                  <p>{t('fishDataset.tr_passes_estimate', { passes: done.training.estimate.passes })}</p>
                </>
              )}
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
              {onTrain && done.formats?.includes('omnivoice') && (
                <Button size="xs" onClick={() => onTrain(done.path)}>
                  <CpuIcon /> {t('fishTraining.train_open')}
                </Button>
              )}
            </div>
          )}
        </div>

        {tooManyTags && (
          <p role="alert" className="text-[11px] leading-snug text-amber-500">
            {t('fishDataset.tags_too_many', { max: MAX_KEPT_TAGS, count: keptEntries.length })}
          </p>
        )}

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('fishStory.done')}</DialogClose>
          <Button disabled={
              busy || !dest.trim() || formats.length === 0 || clips === 0 || unknown.length > 0 || tooManyTags || trainingInvalid || qwenBlocked
            } onClick={() => void run()}>
            {busy && <LoaderCircleIcon className="animate-spin" />} {t('fishDataset.export_button')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
