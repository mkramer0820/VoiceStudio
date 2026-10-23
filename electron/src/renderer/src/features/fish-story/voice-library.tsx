import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { CheckIcon, CopyIcon, ExternalLinkIcon, PlusIcon, Trash2Icon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import type { FishVoice } from '@/features/settings/fish-voice-picker';
import type { VoiceRecord } from './use-fish-story';

function when(seconds: number | null): string {
  return seconds ? new Date(seconds * 1000).toLocaleDateString() : '';
}

/** One saved voice: your name for the character first, then everything Fish told us. */
function VoiceRow({
  record,
  onBoard,
  onAdd,
  onRename,
  onNotes,
  onForget,
}: {
  record: VoiceRecord;
  onBoard: boolean;
  onAdd: (id: string, options?: { name?: string; fish?: FishVoice }) => void;
  onRename: (id: string, alias: string) => void;
  onNotes: (id: string, notes: string) => void;
  onForget: (id: string) => void;
}) {
  const { t } = useTranslation();
  const [alias, setAlias] = useState<string | null>(null);
  const [notes, setNotes] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const details = [record.author, record.languages.join(', '), record.visibility, record.state]
    .filter(Boolean)
    .join(' · ');

  const commitAlias = () => {
    if (alias !== null && alias.trim() !== record.alias) onRename(record.voice_id, alias.trim());
    setAlias(null);
  };
  const commitNotes = () => {
    if (notes !== null && notes.trim() !== record.notes) onNotes(record.voice_id, notes.trim());
    setNotes(null);
  };

  return (
    <li data-testid="library-row" className="space-y-1.5 rounded-lg border border-border/50 p-2 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          aria-label={t('fishStory.lib_alias')}
          className="h-8 w-52 font-medium"
          placeholder={record.fish_name || t('fishStory.lib_alias_placeholder')}
          value={alias ?? record.alias}
          onChange={(e) => setAlias(e.target.value)}
          onBlur={commitAlias}
          onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        />
        {record.fish_name && (
          <span className="text-xs text-muted-foreground">
            {t('fishStory.lib_on_fish')}: <b className="text-foreground/80">{record.fish_name}</b>
          </span>
        )}
        <Button
          size="xs"
          variant={onBoard ? 'ghost' : 'outline'}
          disabled={onBoard}
          aria-label={t('fishStory.voice_add_named', { name: record.alias || record.fish_name || record.voice_id })}
          className="ml-auto"
          onClick={() => onAdd(record.voice_id)}
        >
          {onBoard ? <CheckIcon /> : <PlusIcon />}
          {onBoard ? t('fishStory.lib_on_board') : t('fishStory.lib_use')}
        </Button>
        <Button
          size="icon-sm"
          variant="ghost"
          aria-label={t('fishStory.lib_forget', { name: record.alias || record.fish_name || record.voice_id })}
          onClick={() => onForget(record.voice_id)}
        >
          <Trash2Icon />
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
        <code className="rounded bg-muted/50 px-1.5 py-0.5 font-mono select-all">{record.voice_id}</code>
        <Button
          size="icon-sm"
          variant="ghost"
          aria-label={t('fishStory.lib_copy_id')}
          onClick={() => {
            void navigator.clipboard?.writeText(record.voice_id).then(() => setCopied(true));
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
        </Button>
        <a
          href={record.fish_url}
          target="_blank"
          rel="noreferrer noopener"
          className="inline-flex items-center gap-1 underline-offset-2 hover:underline"
        >
          <ExternalLinkIcon className="size-3" /> {t('fishStory.lib_open_fish')}
        </a>
        {record.uses > 0 && (
          <span>
            {t('fishStory.lib_used', { lines: record.lines, runs: record.uses, date: when(record.last_used_at) })}
          </span>
        )}
      </div>

      {(details || record.tags.length > 0 || record.description) && (
        <div className="space-y-0.5 text-[11px] text-muted-foreground">
          {details && <p>{details}</p>}
          {record.tags.length > 0 && <p>{record.tags.slice(0, 8).join(', ')}</p>}
          {record.description && <p className="line-clamp-2">{record.description}</p>}
        </div>
      )}

      <Input
        aria-label={t('fishStory.lib_notes')}
        className="h-7 text-xs"
        placeholder={t('fishStory.lib_notes')}
        value={notes ?? record.notes}
        onChange={(e) => setNotes(e.target.value)}
        onBlur={commitNotes}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
      />
    </li>
  );
}

/**
 * Every voice you have saved, listed or rendered with, searchable by your character name, Fish's
 * name or the id. Renaming here renames the character on the story board too.
 */
export function VoiceLibrary({
  library,
  voiceNames,
  onAdd,
  onRename,
  onNotes,
  onForget,
}: {
  library: VoiceRecord[];
  voiceNames: Record<string, string>;
  onAdd: (id: string, options?: { name?: string; fish?: FishVoice }) => void;
  onRename: (id: string, alias: string) => void;
  onNotes: (id: string, notes: string) => void;
  onForget: (id: string) => void;
}) {
  const { t } = useTranslation();
  const [query, setQuery] = useState('');
  const needle = query.trim().toLowerCase();
  const rows = needle
    ? library.filter((r) =>
        [r.alias, r.fish_name, r.voice_id, r.notes, r.author ?? '', r.tags.join(' '), r.description]
          .join(' ')
          .toLowerCase()
          .includes(needle),
      )
    : library;

  return (
    <section className="space-y-2">
      <h3 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        {t('fishStory.lib_title', { count: library.length })}
      </h3>
      <p className="text-xs text-muted-foreground">{t('fishStory.lib_note')}</p>
      {library.length > 0 && (
        <Input
          aria-label={t('fishStory.lib_search')}
          placeholder={t('fishStory.lib_search')}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      )}
      {library.length === 0 ? (
        <p className="text-xs text-muted-foreground">{t('fishStory.lib_empty')}</p>
      ) : rows.length === 0 ? (
        <p className="text-xs text-muted-foreground">{t('fishStory.library_none')}</p>
      ) : (
        <ul className="max-h-80 space-y-1.5 overflow-y-auto" data-testid="library-list">
          {rows.map((record) => (
            <VoiceRow
              key={record.voice_id}
              record={record}
              onBoard={record.voice_id in voiceNames}
              onAdd={onAdd}
              onRename={onRename}
              onNotes={onNotes}
              onForget={onForget}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
