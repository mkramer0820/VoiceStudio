import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { actionOf, parseDictionary, renderTag, writtenAs, type TagAction, type TagActions } from './export-options';

const FILTER_AT = 12;
const selectClass =
  'h-7 rounded-md border border-input bg-input/20 px-1 text-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

/** One row per unique tag in the selection: remove it (default), keep it, or write it another way. */
export function TagsSection({
  knownTags,
  actions,
  onChange,
  unknown,
  omnivoiceTags,
  checkKnown,
}: {
  knownTags: Array<[string, number]>;
  actions: TagActions;
  onChange: (actions: TagActions) => void;
  /** Tags the OmniVoice tag set does not include (empty when not checking). */
  unknown: string[];
  omnivoiceTags: string[];
  checkKnown: boolean;
}) {
  const { t } = useTranslation();
  const [filter, setFilter] = useState('');
  const [paste, setPaste] = useState('');
  const tags = knownTags.map(([tag]) => tag);

  if (knownTags.length === 0) {
    return <p className="text-[11px] text-muted-foreground">{t('fishDataset.tags_none')}</p>;
  }

  const set = (tag: string, action: TagAction) => onChange({ ...actions, [tag]: action });
  const shown = knownTags.filter(([tag]) => tag.includes(filter.trim().toLowerCase()));

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium">{t('fishDataset.tags_title')}</span>
        <span className="flex gap-1">
          <Button size="xs" variant="outline" onClick={() => onChange({})}>
            {t('fishDataset.tags_remove_all')}
          </Button>
          <Button
            size="xs"
            variant="outline"
            onClick={() => onChange(Object.fromEntries(tags.map((tag) => [tag, { mode: 'keep', to: '' }])))}
          >
            {t('fishDataset.tags_keep_all')}
          </Button>
        </span>
      </div>
      <p className="text-[11px] text-muted-foreground">{t('fishDataset.tags_hint')}</p>
      {knownTags.length > FILTER_AT && (
        <Input
          aria-label={t('fishDataset.tags_filter')}
          placeholder={t('fishDataset.tags_filter')}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      )}
      <ul className="max-h-40 space-y-1 overflow-y-auto rounded-md border border-border/50 p-1.5" aria-label={t('fishDataset.tags_title')}>
        {shown.map(([tag, count]) => {
          const action = actionOf(actions, tag);
          const flagged = unknown.includes(tag);
          return (
            <li key={tag} data-testid="tag-row" className="flex items-center gap-2 text-xs">
              <span className="w-32 shrink-0 truncate font-mono" title={`[${tag}]`}>
                [{tag}] <span className="text-muted-foreground">{count}</span>
              </span>
              <select
                aria-label={t('fishDataset.tag_action', { tag })}
                className={selectClass}
                value={action.mode}
                onChange={(e) => set(tag, { mode: e.target.value as TagAction['mode'], to: action.to })}
              >
                <option value="remove">{t('fishDataset.tag_remove')}</option>
                <option value="keep">{t('fishDataset.tag_keep')}</option>
                <option value="convert">{t('fishDataset.tag_convert')}</option>
              </select>
              {action.mode === 'convert' && (
                <Input
                  aria-label={t('fishDataset.tag_convert_to', { tag })}
                  className={`h-7 flex-1 font-mono text-xs ${flagged ? 'border-amber-500 text-amber-500' : ''}`}
                  placeholder="sad"
                  value={action.to}
                  onChange={(e) => set(tag, { mode: 'convert', to: e.target.value })}
                />
              )}
              {action.mode !== 'convert' && flagged && (
                <span className="text-[11px] text-amber-500">{t('fishDataset.tag_unknown_short')}</span>
              )}
            </li>
          );
        })}
      </ul>
      {unknown.length > 0 && checkKnown && (
        <p role="alert" className="text-[11px] leading-snug text-amber-500">
          {t('fishDataset.tags_unknown', {
            tags: unknown.map((tag) => renderTag(writtenAs(tag, actionOf(actions, tag)) ?? tag)).join(', '),
            known: omnivoiceTags.join(', '),
          })}
        </p>
      )}
      <details className="text-[11px]">
        <summary className="cursor-pointer text-muted-foreground">{t('fishDataset.tags_paste')}</summary>
        <textarea
          aria-label={t('fishDataset.tags_paste')}
          className="mt-1 h-16 w-full rounded-md border border-input bg-input/20 p-1.5 font-mono text-xs dark:bg-input/30"
          placeholder="[frustrated]:(sad)&#10;laughing=laughter"
          value={paste}
          onChange={(e) => setPaste(e.target.value)}
        />
        <Button size="xs" variant="outline" onClick={() => onChange({ ...actions, ...parseDictionary(paste, tags) })}>
          {t('fishDataset.tags_paste_apply')}
        </Button>
      </details>
    </div>
  );
}
