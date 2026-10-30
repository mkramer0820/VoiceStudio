import { useTranslation } from 'react-i18next';
import { AlertTriangleIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { EmotionPicker, selectClass } from './emotion-picker';
import { characterRows, type Draft, type Emotion, type EmotionCatalog, type MappingWarning, type Profile } from './story-model';

/** Character -> profile mapping and a default emotion per character. Free mapping: warnings, never blocks. */
export function CharacterTable({
  draft,
  profiles,
  catalog,
  warnings,
  onNarrator,
  onCast,
  onEmotion,
  onSelectLines,
}: {
  /** Tick every line of a character (null = narration), to apply an emotion or speaker to all of them. */
  onSelectLines: (name: string | null) => void;
  draft: Draft;
  profiles: Profile[];
  catalog: EmotionCatalog | null;
  warnings: MappingWarning[];
  onNarrator: (id: string | null) => void;
  onCast: (name: string, id: string | null) => void;
  onEmotion: (name: string | null, emotion: Emotion | null) => void;
}) {
  const { t } = useTranslation();
  const rows = characterRows(draft.chunks);
  const choose = (value: string | null, set: (id: string | null) => void, label: string) => (
    <select aria-label={label} className={`${selectClass} w-48`} value={value ?? ''} onChange={(e) => set(e.target.value || null)}>
      <option value="">{t('qwenStory.profile_none')}</option>
      {profiles.map((p) => (
        <option key={p.id} value={p.id}>
          {p.name}
        </option>
      ))}
    </select>
  );
  return (
    <section aria-label={t('qwenStory.characters_title')} className="space-y-2 rounded-xl border border-border/60 bg-card/30 p-3">
      <h2 className="text-sm font-medium">{t('qwenStory.characters_title')}</h2>
      {rows.length === 0 && <p className="text-xs text-muted-foreground">{t('qwenStory.characters_empty')}</p>}
      <table className="w-full text-xs">
        <tbody>
          {rows.map((row) => {
            const label = row.name ?? t('qwenStory.narration');
            return (
              <tr key={row.name ?? '__narration'} className="border-t border-border/30">
                <td className="py-1.5 pr-2 font-medium">
                  {label}
                  <span className="ml-2 font-normal text-muted-foreground">{t('qwenStory.chunks_count', { count: row.chunks })}</span>
                </td>
                <td className="py-1.5 pr-2">
                  {row.name === null
                    ? choose(draft.narrator_profile_id, onNarrator, t('qwenStory.profile_for', { name: label }))
                    : choose(draft.cast[row.name] ?? null, (id) => onCast(row.name as string, id), t('qwenStory.profile_for', { name: label }))}
                </td>
                <td className="py-1.5 pr-2">
                  <Button size="xs" variant="ghost" aria-label={t('qwenStory.select_lines_of', { name: label })} onClick={() => onSelectLines(row.name)}>
                    {t('qwenStory.select_lines')}
                  </Button>
                </td>
                <td className="py-1.5">
                  {row.name !== null && (
                    <EmotionPicker
                      catalog={catalog}
                      allowInherit
                      label={t('qwenStory.emotion_for', { name: label })}
                      value={draft.emotions[row.name] ?? null}
                      onChange={(e) => onEmotion(row.name, e)}
                    />
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {warnings.map((w) => (
        <p key={w.key + JSON.stringify(w.values)} role="alert" className="flex items-start gap-1 text-xs text-amber-500">
          <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
          {t(w.key, w.values)}
        </p>
      ))}
    </section>
  );
}
