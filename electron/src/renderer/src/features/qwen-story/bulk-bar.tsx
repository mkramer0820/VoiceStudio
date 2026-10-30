import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PlayIcon, Trash2Icon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { EmotionPicker, selectClass } from './emotion-picker';
import type { Emotion, EmotionCatalog } from './story-model';

/** Actions on the ticked chunks: the same idea as the Fish board (select lines, then apply to all of them). */
export function BulkBar({
  count,
  total,
  speakers,
  catalog,
  running,
  onSelectAll,
  onClear,
  onEmotion,
  onSpeaker,
  onRender,
  onRemove,
}: {
  count: number;
  total: number;
  speakers: string[];
  catalog: EmotionCatalog | null;
  running: boolean;
  onSelectAll: () => void;
  onClear: () => void;
  onEmotion: (emotion: Emotion | null) => void;
  onSpeaker: (speaker: string | null) => void;
  onRender: () => void;
  onRemove: () => void;
}) {
  const { t } = useTranslation();
  const [emotion, setEmotion] = useState<Emotion | null>(null);
  return (
    <div role="toolbar" aria-label={t('qwenStory.bulk_title')} className="flex flex-wrap items-center gap-2 rounded-xl border border-border/60 bg-card/30 p-2 text-xs">
      <span className="font-medium">{t('qwenStory.selected_count', { count })}</span>
      <Button size="xs" variant="outline" disabled={count === total} onClick={onSelectAll}>
        {t('qwenStory.select_all')}
      </Button>
      <Button size="xs" variant="ghost" disabled={count === 0} onClick={onClear}>
        {t('qwenStory.select_none')}
      </Button>
      <span className="inline-flex items-center gap-1">
        <EmotionPicker catalog={catalog} allowInherit label={t('qwenStory.bulk_emotion')} value={emotion} onChange={setEmotion} />
        <Button size="xs" variant="outline" disabled={count === 0} onClick={() => onEmotion(emotion)}>
          {emotion ? t('qwenStory.bulk_apply_emotion') : t('qwenStory.bulk_clear_emotion')}
        </Button>
      </span>
      <select
        aria-label={t('qwenStory.bulk_speaker')}
        className={`${selectClass} w-36`}
        disabled={count === 0}
        value=""
        onChange={(e) => {
          if (e.target.value === '__narration') onSpeaker(null);
          else if (e.target.value) onSpeaker(e.target.value);
        }}
      >
        <option value="">{t('qwenStory.bulk_speaker')}</option>
        <option value="__narration">{t('qwenStory.narration')}</option>
        {speakers.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
      <Button size="xs" disabled={count === 0 || running} onClick={onRender}>
        <PlayIcon /> {t('qwenStory.render_selected')}
      </Button>
      <Button size="xs" variant="ghost" disabled={count === 0 || running} onClick={onRemove}>
        <Trash2Icon /> {t('qwenStory.remove_selected')}
      </Button>
    </div>
  );
}
