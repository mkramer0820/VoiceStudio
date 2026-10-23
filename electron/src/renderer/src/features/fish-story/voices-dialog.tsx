import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { CheckIcon, LoaderCircleIcon, PlusIcon, SearchIcon } from 'lucide-react';
import { VoiceLibrary } from './voice-library';
import type { VoiceRecord } from './use-fish-story';
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
import type { FishVoice } from '@/features/settings/fish-voice-picker';

const ID_RE = /^[A-Za-z0-9_-]{1,128}$/;

/** Pull a Fish voice id out of a pasted id or a fish.audio/m/<id> link. */
export function parseVoiceId(raw: string): string | null {
  const text = raw.trim();
  const match = text.match(/\/m\/([A-Za-z0-9_-]+)/);
  const id = match ? match[1] : text;
  return ID_RE.test(id) ? id : null;
}

/**
 * Put voices on the board that are not in "my voices": by pasting a voice id or
 * link, or by searching the public Voice Library. Searching contacts Fish, only
 * when the button is pressed; adding by id is local.
 */
export function VoicesDialog({
  open,
  onOpenChange,
  voiceNames,
  library,
  onOpen,
  onAdd,
  onRename,
  onNotes,
  onForget,
  onSearch,
  onImportCollection,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  voiceNames: Record<string, string>;
  library: VoiceRecord[];
  onOpen: () => void;
  onAdd: (id: string, options?: { name?: string; fish?: FishVoice }) => void;
  onRename: (id: string, alias: string) => void;
  onNotes: (id: string, notes: string) => void;
  onForget: (id: string) => void;
  onSearch: (title: string, language: string) => Promise<FishVoice[] | null>;
  onImportCollection: (url: string) => Promise<{ title: string; voices: FishVoice[] } | null>;
}) {
  const { t } = useTranslation();
  const [rawId, setRawId] = useState('');
  const [name, setName] = useState('');
  const [title, setTitle] = useState('');
  const [language, setLanguage] = useState('');
  const [results, setResults] = useState<FishVoice[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [collectionUrl, setCollectionUrl] = useState('');
  const [collection, setCollection] = useState<{ title: string; voices: FishVoice[] } | null>(null);
  const [importing, setImporting] = useState(false);
  const parsed = parseVoiceId(rawId);

  useEffect(() => {
    if (open) onOpen();
  }, [open, onOpen]);

  const search = async () => {
    setBusy(true);
    setResults(await onSearch(title, language));
    setBusy(false);
  };

  const importCollection = async () => {
    setImporting(true);
    setCollection(await onImportCollection(collectionUrl));
    setImporting(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t('fishStory.voices_title')}</DialogTitle>
          <DialogDescription>{t('fishStory.voices_desc')}</DialogDescription>
        </DialogHeader>

        <VoiceLibrary
          library={library}
          voiceNames={voiceNames}
          onAdd={onAdd}
          onRename={onRename}
          onNotes={onNotes}
          onForget={onForget}
        />

        <section className="space-y-2">
          <h3 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            {t('fishStory.voices_by_id')}
          </h3>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              aria-label={t('fishStory.voice_id')}
              className="w-72 font-mono text-xs"
              placeholder={t('fishStory.voice_id_placeholder')}
              value={rawId}
              onChange={(e) => setRawId(e.target.value)}
            />
            <Input
              aria-label={t('fishStory.voice_name')}
              className="w-44"
              placeholder={t('fishStory.voice_name')}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <Button
              size="sm"
              disabled={!parsed}
              onClick={() => {
                if (!parsed) return;
                onAdd(parsed, { name });
                setRawId('');
                setName('');
              }}
            >
              <PlusIcon /> {t('fishStory.voice_add')}
            </Button>
          </div>
          {rawId.trim() && !parsed && (
            <p role="alert" className="text-xs text-amber-500">
              {t('fishStory.voice_id_invalid')}
            </p>
          )}
        </section>

        <section className="space-y-2">
          <h3 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            {t('fishStory.voices_collection')}
          </h3>
          <p className="text-xs text-muted-foreground">{t('fishStory.voices_collection_note')}</p>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              aria-label={t('fishStory.collection_url')}
              className="w-96 max-w-full font-mono text-xs"
              placeholder="https://fish.audio/collections/…"
              value={collectionUrl}
              onChange={(e) => setCollectionUrl(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && collectionUrl.trim() && void importCollection()}
            />
            <Button
              size="sm"
              variant="outline"
              disabled={importing || !collectionUrl.trim()}
              onClick={() => void importCollection()}
            >
              {importing ? <LoaderCircleIcon className="animate-spin" /> : <SearchIcon />}
              {t('fishStory.collection_import')}
            </Button>
          </div>
          {collection && (
            <div className="space-y-1" data-testid="collection-results">
              <div className="flex items-center gap-2">
                <p className="min-w-0 flex-1 truncate text-sm font-medium">
                  {collection.title || t('fishStory.collection_untitled')}
                </p>
                <Button
                  size="xs"
                  disabled={collection.voices.every((v) => v.id in voiceNames)}
                  onClick={() => collection.voices.forEach((v) => onAdd(v.id, { fish: v }))}
                >
                  <PlusIcon /> {t('fishStory.collection_add_all', { count: collection.voices.length })}
                </Button>
              </div>
              <ul className="space-y-1">
                {collection.voices.map((voice) => {
                  const added = voice.id in voiceNames;
                  return (
                    <li
                      key={voice.id}
                      className="flex items-center gap-2 rounded-lg border border-border/50 px-2 py-1.5 text-sm"
                    >
                      <span className="min-w-0 flex-1 truncate">{voice.title || voice.id}</span>
                      <Button
                        size="xs"
                        variant="outline"
                        disabled={added}
                        aria-label={t('fishStory.voice_add_named', { name: voice.title || voice.id })}
                        onClick={() => onAdd(voice.id, { fish: voice })}
                      >
                        {added ? <CheckIcon /> : <PlusIcon />}
                        {added ? t('fishStory.voice_added') : t('fishStory.voice_add')}
                      </Button>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </section>

        <section className="space-y-2">
          <h3 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            {t('fishStory.voices_library')}
          </h3>
          <p className="text-xs text-muted-foreground">{t('fishStory.voices_library_note')}</p>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              aria-label={t('fishStory.library_search')}
              className="w-56"
              placeholder={t('fishStory.library_search')}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && void search()}
            />
            <Input
              aria-label={t('fishStory.library_language')}
              className="w-28"
              placeholder={t('fishStory.library_language')}
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && void search()}
            />
            <Button size="sm" variant="outline" disabled={busy} onClick={() => void search()}>
              {busy ? <LoaderCircleIcon className="animate-spin" /> : <SearchIcon />}
              {t('fishStory.library_go')}
            </Button>
          </div>
          {results && results.length === 0 && (
            <p className="text-xs text-muted-foreground">{t('fishStory.library_none')}</p>
          )}
          {results && results.length > 0 && (
            <ul className="max-h-64 space-y-1 overflow-y-auto" data-testid="library-results">
              {results.map((voice) => {
                const added = voice.id in voiceNames;
                return (
                  <li
                    key={voice.id}
                    className="flex items-center gap-2 rounded-lg border border-border/50 px-2 py-1.5 text-sm"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium">{voice.title || voice.id}</p>
                      <p className="truncate text-[11px] text-muted-foreground">
                        {[voice.author, voice.languages?.join(', '), voice.tags?.slice(0, 4).join(', ')]
                          .filter(Boolean)
                          .join(' · ')}
                      </p>
                    </div>
                    <Button
                      size="xs"
                      variant="outline"
                      disabled={added}
                      aria-label={t('fishStory.voice_add_named', { name: voice.title || voice.id })}
                      onClick={() => onAdd(voice.id, { fish: voice })}
                    >
                      {added ? <CheckIcon /> : <PlusIcon />}
                      {added ? t('fishStory.voice_added') : t('fishStory.voice_add')}
                    </Button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('fishStory.done')}</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
