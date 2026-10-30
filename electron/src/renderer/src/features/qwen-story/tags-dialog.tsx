import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import {
  characterRows,
  clearEmotions,
  countMatches,
  emotionCounts,
  findReplace,
  renameCharacter,
  type Draft,
} from './story-model';

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2 border-t border-border/50 pt-3 first:border-t-0 first:pt-0">
      <h3 className="text-sm font-medium">{title}</h3>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      {children}
    </section>
  );
}

/** Story-wide tag tools, like the Fish tags dialog: characters, emotion tags, find and replace. */
export function TagsDialog({
  open,
  onOpenChange,
  draft,
  onDraft,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  draft: Draft;
  onDraft: (next: Draft) => void;
}) {
  const { t } = useTranslation();
  const [renames, setRenames] = useState<Record<string, string>>({});
  const [find, setFind] = useState('');
  const [replace, setReplace] = useState('');
  const [matchCase, setMatchCase] = useState(false);
  const characters = characterRows(draft.chunks).filter((r) => r.name !== null);
  const emotions = emotionCounts(draft.chunks);
  const found = useMemo(() => countMatches(draft.chunks, find, matchCase), [draft.chunks, find, matchCase]);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t('qwenStory.tags_title')}</DialogTitle>
          <DialogDescription>{t('qwenStory.tags_desc')}</DialogDescription>
        </DialogHeader>
        <div className="space-y-4 text-sm">
          <Section title={t('qwenStory.tags_characters')} hint={t('qwenStory.tags_characters_hint')}>
            {characters.length === 0 && <p className="text-xs text-muted-foreground">{t('qwenStory.characters_empty')}</p>}
            {characters.map((row) => (
              <div key={row.name} className="flex items-center gap-2">
                <span className="w-32 truncate text-xs font-medium">{row.name}</span>
                <Input
                  aria-label={t('qwenStory.tags_rename', { name: row.name as string })}
                  className="h-8 w-44"
                  value={renames[row.name as string] ?? ''}
                  placeholder={t('qwenStory.tags_new_name')}
                  onChange={(e) => setRenames({ ...renames, [row.name as string]: e.target.value })}
                />
                <Button
                  size="xs"
                  variant="outline"
                  disabled={!(renames[row.name as string] ?? '').trim()}
                  onClick={() => {
                    onDraft(renameCharacter(draft, row.name as string, renames[row.name as string]));
                    setRenames({ ...renames, [row.name as string]: '' });
                  }}
                >
                  {t('qwenStory.tags_rename_apply')}
                </Button>
              </div>
            ))}
          </Section>
          <Section title={t('qwenStory.tags_emotions')} hint={t('qwenStory.tags_emotions_hint')}>
            {emotions.length === 0 && <p className="text-xs text-muted-foreground">{t('qwenStory.tags_no_emotions')}</p>}
            {emotions.map((e) => (
              <div key={e.preset} className="flex items-center gap-2 text-xs">
                <span className="w-32 font-medium">{t(`qwenStory.emotion_${e.preset}`)}</span>
                <span className="text-muted-foreground">{t('qwenStory.chunks_count', { count: e.count })}</span>
                <Button size="xs" variant="outline" onClick={() => onDraft({ ...draft, chunks: clearEmotions(draft.chunks, e.preset) })}>
                  {t('qwenStory.tags_clear_one')}
                </Button>
              </div>
            ))}
            {emotions.length > 1 && (
              <Button size="xs" variant="outline" onClick={() => onDraft({ ...draft, chunks: clearEmotions(draft.chunks) })}>
                {t('qwenStory.tags_clear_all')}
              </Button>
            )}
          </Section>
          <Section title={t('qwenStory.tags_replace')}>
            <div className="flex flex-wrap items-center gap-2">
              <Input aria-label={t('qwenStory.tags_find')} className="h-8 w-44" value={find} placeholder={t('qwenStory.tags_find')} onChange={(e) => setFind(e.target.value)} />
              <Input aria-label={t('qwenStory.tags_replace_with')} className="h-8 w-44" value={replace} placeholder={t('qwenStory.tags_replace_with')} onChange={(e) => setReplace(e.target.value)} />
              <label className="flex items-center gap-1 text-xs">
                <input type="checkbox" checked={matchCase} onChange={(e) => setMatchCase(e.target.checked)} />
                {t('qwenStory.tags_match_case')}
              </label>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">{find ? t('qwenStory.tags_matches', { matches: found.matches, lines: found.lines }) : ''}</span>
              <Button size="xs" variant="outline" disabled={found.matches === 0} onClick={() => onDraft({ ...draft, chunks: findReplace(draft.chunks, find, replace, matchCase) })}>
                {t('qwenStory.tags_replace_all')}
              </Button>
            </div>
          </Section>
        </div>
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('qwenStory.close')}</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
