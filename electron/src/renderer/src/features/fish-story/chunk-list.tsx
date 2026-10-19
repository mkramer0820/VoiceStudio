import { memo, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowDownToLineIcon, MergeIcon, PlusIcon, Trash2Icon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { StoryChunk } from './story-model';

interface RowProps {
  index: number;
  chunk: StoryChunk;
  selected: boolean;
  /** Display name of the assigned voice, or '' when the default is used. */
  voiceName: string;
  hue: number | null;
  last: boolean;
  onSelect: (id: string, mode: 'single' | 'range' | 'toggle') => void;
  onEdit: (id: string, text: string) => void;
  onMerge: (id: string) => void;
  onInsert: (id: string) => void;
  onRemove: (id: string) => void;
}

const ChunkRow = memo(function ChunkRow({
  index,
  chunk,
  selected,
  voiceName,
  hue,
  last,
  onSelect,
  onEdit,
  onMerge,
  onInsert,
  onRemove,
}: RowProps) {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(chunk.text);
  const area = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!editing) setText(chunk.text);
  }, [chunk.text, editing]);
  useEffect(() => {
    if (editing) area.current?.focus();
  }, [editing]);

  const commit = () => {
    setEditing(false);
    if (text !== chunk.text) onEdit(chunk.id, text);
  };

  return (
    <li
      data-testid="story-row"
      aria-selected={selected}
      className={cn(
        'group flex items-start gap-3 border-b border-border/40 px-3 py-2 text-sm transition-colors',
        selected ? 'bg-primary/10' : 'hover:bg-accent/30',
      )}
      style={hue === null ? undefined : { boxShadow: `inset 3px 0 0 hsl(${hue} 70% 55%)` }}
      onClick={(event) => {
        if ((event.target as HTMLElement).closest('button,textarea,input')) return;
        onSelect(
          chunk.id,
          event.shiftKey ? 'range' : event.ctrlKey || event.metaKey ? 'toggle' : 'single',
        );
      }}
      onDoubleClick={() => setEditing(true)}
    >
      <input
        type="checkbox"
        aria-label={t('fishStory.select_line', { n: index })}
        checked={selected}
        className="mt-1 size-4 shrink-0 accent-[hsl(var(--primary))]"
        onChange={() => onSelect(chunk.id, 'toggle')}
      />
      <span className="mt-0.5 w-9 shrink-0 text-right font-mono text-xs text-muted-foreground tabular-nums">
        {index}
      </span>
      <span
        className="mt-0.5 w-28 shrink-0 truncate rounded-md px-2 py-0.5 text-xs"
        style={
          hue === null
            ? undefined
            : { background: `hsl(${hue} 70% 55% / 0.16)`, color: `hsl(${hue} 80% 70%)` }
        }
        title={voiceName || t('fishStory.default_voice_name')}
      >
        {voiceName || <span className="text-muted-foreground">{t('fishStory.default_voice_name')}</span>}
      </span>
      <div className="min-w-0 flex-1">
        {editing ? (
          <textarea
            ref={area}
            value={text}
            rows={Math.min(8, Math.max(2, text.split('\n').length))}
            aria-label={t('fishStory.edit_line', { n: index })}
            className="w-full resize-y rounded-md border border-border bg-background/60 p-2 text-sm"
            onChange={(event) => setText(event.target.value)}
            onBlur={commit}
            onKeyDown={(event) => {
              if (event.key === 'Escape') {
                setText(chunk.text);
                setEditing(false);
              } else if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) commit();
            }}
          />
        ) : (
          <p className="whitespace-pre-wrap break-words">{chunk.text}</p>
        )}
      </div>
      <span className="mt-0.5 w-12 shrink-0 text-right font-mono text-xs text-muted-foreground tabular-nums">
        {chunk.text.trim().length}
      </span>
      <span className="flex shrink-0 gap-0.5 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label={t('fishStory.insert_below', { n: index })}
          onClick={() => onInsert(chunk.id)}
        >
          <PlusIcon />
        </Button>
        <Button
          variant="ghost"
          size="icon-xs"
          disabled={last}
          aria-label={t('fishStory.merge_next', { n: index })}
          onClick={() => onMerge(chunk.id)}
        >
          <MergeIcon />
        </Button>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label={t('fishStory.delete_line', { n: index })}
          onClick={() => onRemove(chunk.id)}
        >
          <Trash2Icon />
        </Button>
      </span>
    </li>
  );
});

export function ChunkList({
  chunks,
  selected,
  voiceNames,
  hueFor,
  onSelect,
  onEdit,
  onMerge,
  onInsert,
  onRemove,
}: {
  chunks: StoryChunk[];
  selected: ReadonlySet<string>;
  voiceNames: Record<string, string>;
  hueFor: (voice: string | null | undefined) => number | null;
  onSelect: RowProps['onSelect'];
  onEdit: RowProps['onEdit'];
  onMerge: RowProps['onMerge'];
  onInsert: RowProps['onInsert'];
  onRemove: RowProps['onRemove'];
}) {
  const { t } = useTranslation();
  if (chunks.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 p-10 text-center text-sm text-muted-foreground">
        <ArrowDownToLineIcon className="size-6" />
        <p>{t('fishStory.empty')}</p>
      </div>
    );
  }
  return (
    <ol aria-label={t('fishStory.lines')} className="select-none">
      {chunks.map((chunk, i) => (
        <ChunkRow
          key={chunk.id}
          index={i + 1}
          chunk={chunk}
          selected={selected.has(chunk.id)}
          voiceName={chunk.voice ? (voiceNames[chunk.voice] ?? chunk.voice) : ''}
          hue={hueFor(chunk.voice)}
          last={i === chunks.length - 1}
          onSelect={onSelect}
          onEdit={onEdit}
          onMerge={onMerge}
          onInsert={onInsert}
          onRemove={onRemove}
        />
      ))}
    </ol>
  );
}
