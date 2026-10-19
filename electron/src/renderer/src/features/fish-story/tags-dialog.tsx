import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { StoryChunk } from './story-model';
import {
  applyCast,
  countMatches,
  detectSpeakers,
  extractSpeakers,
  findReplace,
  rewriteTags,
  scanTags,
  type DirectionAction,
  type TagKinds,
} from './tag-tools';

const selectClass =
  'h-8 min-w-0 rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

type Rule = { action: 'keep' | 'remove' | 'replace'; to: string };

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2 border-t border-border/50 pt-3 first:border-t-0 first:pt-0">
      <h3 className="text-sm font-medium">{title}</h3>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      {children}
    </section>
  );
}

export function TagsDialog({
  open,
  onOpenChange,
  chunks,
  voiceNames,
  cast,
  narrator,
  inheritedNarrator,
  onChunks,
  onCast,
  onNarrator,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  chunks: StoryChunk[];
  voiceNames: Record<string, string>;
  cast: Record<string, string>;
  /** The story's default voice: narration and any line without a speaker. */
  narrator: string | null;
  /** The default from Settings (shown as the "default" choice). */
  inheritedNarrator: string | null;
  onChunks: (next: StoryChunk[]) => void;
  onCast: (next: Record<string, string>) => void;
  onNarrator: (voice: string | null) => void;
}) {
  const { t } = useTranslation();
  const [kinds, setKinds] = useState<TagKinds>({});
  const [strip, setStrip] = useState(true);
  const [carry, setCarry] = useState(false);
  const [rules, setRules] = useState<Record<string, Rule>>({});
  const [find, setFind] = useState('');
  const [replace, setReplace] = useState('');
  const [matchCase, setMatchCase] = useState(false);
  const [notice, setNotice] = useState('');

  const tags = useMemo(() => scanTags(chunks, kinds), [chunks, kinds]);
  const speakerSet = useMemo(() => detectSpeakers(chunks, kinds), [chunks, kinds]);
  // Speakers still sitting in the text as tags, plus ones already extracted onto lines.
  const speakers = useMemo(() => {
    const rows = new Map<string, { tagged: number; lines: number }>();
    for (const name of speakerSet) rows.set(name, { tagged: 0, lines: 0 });
    for (const tag of tags) if (tag.kind === 'speaker') rows.get(tag.name)!.tagged = tag.count;
    for (const c of chunks) {
      if (!c.speaker) continue;
      const row = rows.get(c.speaker) ?? { tagged: 0, lines: 0 };
      row.lines += 1;
      rows.set(c.speaker, row);
    }
    return [...rows.entries()];
  }, [speakerSet, tags, chunks]);
  const directions = tags.filter((tag) => tag.kind === 'direction');
  const found = useMemo(() => countMatches(chunks, find, matchCase), [chunks, find, matchCase]);

  const applySpeakers = () => {
    const extracted = extractSpeakers(chunks, speakerSet, { carry, strip });
    const mapped = applyCast(extracted, cast);
    onChunks(mapped);
    const assigned = mapped.filter((c) => c.speaker && cast[c.speaker]).length;
    setNotice(t('fishStory.tags_applied_speakers', { lines: assigned }));
  };

  const applyDirections = () => {
    const active: Record<string, DirectionAction> = {};
    for (const [name, rule] of Object.entries(rules)) {
      if (rule.action === 'remove') active[name] = { action: 'remove' };
      else if (rule.action === 'replace') active[name] = { action: 'replace', to: rule.to };
    }
    onChunks(rewriteTags(chunks, active));
    setRules({});
    setNotice(t('fishStory.tags_applied_directions', { count: Object.keys(active).length }));
  };

  const voiceOptions = Object.entries(voiceNames);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t('fishStory.tags_title')}</DialogTitle>
          <DialogDescription>{t('fishStory.tags_desc')}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4 text-sm">
          <Section title={t('fishStory.narrator')} hint={t('fishStory.narrator_hint')}>
            <select
              aria-label={t('fishStory.narrator')}
              className={`${selectClass} w-full sm:w-72`}
              value={narrator ?? ''}
              onChange={(e) => onNarrator(e.target.value || null)}
            >
              <option value="">
                {inheritedNarrator
                  ? t('fishStory.inherit', { value: voiceNames[inheritedNarrator] ?? inheritedNarrator })
                  : t('fishStory.fish_default')}
              </option>
              {voiceOptions.map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
            </select>
          </Section>

          <Section title={t('fishStory.speakers')} hint={t('fishStory.speakers_hint')}>
            {speakers.length === 0 ? (
              <p className="text-xs text-muted-foreground">{t('fishStory.speakers_none')}</p>
            ) : (
              <ul className="space-y-1.5">
                {speakers.map(([name, info]) => (
                  <li key={name} className="flex flex-wrap items-center gap-2">
                    <span className="w-32 truncate font-medium" title={name}>
                      {name}
                    </span>
                    <span className="w-28 text-xs text-muted-foreground tabular-nums">
                      {t('fishStory.speaker_counts', { tags: info.tagged, lines: info.lines })}
                    </span>
                    <select
                      aria-label={t('fishStory.speaker_voice', { name })}
                      className={`${selectClass} w-48`}
                      value={cast[name] ?? ''}
                      onChange={(e) => {
                        const next = { ...cast };
                        if (e.target.value) next[name] = e.target.value;
                        else delete next[name];
                        onCast(next);
                      }}
                    >
                      <option value="">{t('fishStory.not_mapped')}</option>
                      {voiceOptions.map(([id, label]) => (
                        <option key={id} value={id}>
                          {label}
                        </option>
                      ))}
                    </select>
                    {info.tagged > 0 && (
                      <Button
                        size="xs"
                        variant="ghost"
                        onClick={() => setKinds((k) => ({ ...k, [name]: 'direction' }))}
                      >
                        {t('fishStory.not_a_speaker')}
                      </Button>
                    )}
                  </li>
                ))}
              </ul>
            )}
            <div className="flex flex-col gap-1 text-xs">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={strip} onChange={(e) => setStrip(e.target.checked)} />
                {t('fishStory.strip_names')}
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={carry} onChange={(e) => setCarry(e.target.checked)} />
                {t('fishStory.carry_speaker')}
              </label>
            </div>
            <Button size="sm" disabled={speakers.length === 0} onClick={applySpeakers}>
              {t('fishStory.apply_cast')}
            </Button>
          </Section>

          <Section title={t('fishStory.directions')} hint={t('fishStory.directions_hint')}>
            {directions.length === 0 ? (
              <p className="text-xs text-muted-foreground">{t('fishStory.directions_none')}</p>
            ) : (
              <ul className="space-y-1.5">
                {directions.map((tag) => {
                  const rule = rules[tag.name] ?? { action: 'keep', to: '' };
                  return (
                    <li key={tag.name} className="flex flex-wrap items-center gap-2">
                      <code className="w-32 truncate rounded bg-muted/40 px-1.5 py-0.5 text-xs" title={tag.name}>
                        [{tag.name}]
                      </code>
                      <span className="w-12 text-xs text-muted-foreground tabular-nums">×{tag.count}</span>
                      <select
                        aria-label={t('fishStory.direction_action', { name: tag.name })}
                        className={`${selectClass} w-40`}
                        value={rule.action}
                        onChange={(e) =>
                          setRules((r) => ({ ...r, [tag.name]: { ...rule, action: e.target.value as Rule['action'] } }))
                        }
                      >
                        <option value="keep">{t('fishStory.action_keep')}</option>
                        <option value="remove">{t('fishStory.action_remove')}</option>
                        <option value="replace">{t('fishStory.action_replace')}</option>
                      </select>
                      {rule.action === 'replace' && (
                        <Input
                          aria-label={t('fishStory.replace_with')}
                          className="h-8 w-44"
                          placeholder={t('fishStory.replace_with')}
                          value={rule.to}
                          onChange={(e) =>
                            setRules((r) => ({ ...r, [tag.name]: { ...rule, to: e.target.value } }))
                          }
                        />
                      )}
                      <Button
                        size="xs"
                        variant="ghost"
                        onClick={() => setKinds((k) => ({ ...k, [tag.name]: 'speaker' }))}
                      >
                        {t('fishStory.is_a_speaker')}
                      </Button>
                    </li>
                  );
                })}
              </ul>
            )}
            <Button
              size="sm"
              disabled={!Object.values(rules).some((r) => r.action !== 'keep')}
              onClick={applyDirections}
            >
              {t('fishStory.apply_tags')}
            </Button>
          </Section>

          <Section title={t('fishStory.find_replace')} hint={t('fishStory.find_replace_hint')}>
            <div className="flex flex-wrap items-center gap-2">
              <Input
                aria-label={t('fishStory.find')}
                className="h-8 w-48"
                placeholder={t('fishStory.find')}
                value={find}
                onChange={(e) => setFind(e.target.value)}
              />
              <Input
                aria-label={t('fishStory.replace_with')}
                className="h-8 w-48"
                placeholder={t('fishStory.replace_with')}
                value={replace}
                onChange={(e) => setReplace(e.target.value)}
              />
              <label className="flex items-center gap-1 text-xs">
                <input type="checkbox" checked={matchCase} onChange={(e) => setMatchCase(e.target.checked)} />
                {t('fishStory.match_case')}
              </label>
              <Button
                size="sm"
                variant="outline"
                disabled={found.matches === 0}
                onClick={() => {
                  onChunks(findReplace(chunks, find, replace, matchCase));
                  setNotice(t('fishStory.replaced', { count: found.matches }));
                }}
              >
                {t('fishStory.replace_all')}
              </Button>
              <span className="text-xs text-muted-foreground tabular-nums">
                {find ? t('fishStory.matches', { matches: found.matches, lines: found.lines }) : ''}
              </span>
            </div>
          </Section>

          {notice && (
            <p role="status" className="text-xs text-emerald-500">
              {notice}
            </p>
          )}
        </div>

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('fishStory.done')}</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
