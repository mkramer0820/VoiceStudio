import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { LoaderCircleIcon, PlayIcon, SquareIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { apiFetch, describeError } from '@/lib/api/client';

// Audio routes are authenticated, so a plain <audio src> cannot reach them: fetch a blob once per file.
const cache = new Map<string, Promise<string>>();

export function loadRender(session: string, file: string): Promise<string> {
  const key = `${session}/${file}`;
  let hit = cache.get(key);
  if (!hit) {
    hit = (async () => {
      const res = await apiFetch(
        `/api/qwen/playground/sessions/${encodeURIComponent(session)}/audio/${encodeURIComponent(file)}`,
      );
      return URL.createObjectURL(await res.blob());
    })();
    hit.catch(() => cache.delete(key));
    cache.set(key, hit);
  }
  return hit;
}

export interface AbPick {
  session: string;
  file: string;
  label: string;
}

export function RenderPlayer({ session, file }: { session: string; file: string }) {
  const { t } = useTranslation();
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    let alive = true;
    setUrl('');
    setError('');
    loadRender(session, file)
      .then((next) => alive && setUrl(next))
      .catch((reason) => alive && setError(describeError(reason)));
    return () => {
      alive = false;
    };
  }, [session, file]);
  if (error) return <span className="text-xs text-destructive">{error}</span>;
  if (!url) return <LoaderCircleIcon className="size-4 animate-spin text-muted-foreground" aria-label={t('qwenPlayground.loading_audio')} />;
  return <audio controls preload="metadata" src={url} aria-label={file} className="h-8 w-64 max-w-full" />;
}

/** Play render A, then render B, to hear the difference back to back. */
export function AbBar({
  a,
  b,
  onClear,
}: {
  a: AbPick | null;
  b: AbPick | null;
  onClear: () => void;
}) {
  const { t } = useTranslation();
  const [playing, setPlaying] = useState<'a' | 'b' | null>(null);
  const [error, setError] = useState('');
  const current = useRef<HTMLAudioElement | null>(null);
  const token = useRef(0);

  const stop = () => {
    token.current += 1;
    current.current?.pause();
    setPlaying(null);
  };
  useEffect(() => () => {
    token.current += 1;
    current.current?.pause();
  }, []);

  const playOne = async (which: 'a' | 'b', pick: AbPick, mine: number) => {
    const audio = new Audio(await loadRender(pick.session, pick.file));
    if (mine !== token.current) return;
    current.current = audio;
    setPlaying(which);
    await new Promise<void>((resolve, reject) => {
      audio.onended = () => resolve();
      audio.onerror = () => reject(new Error(pick.file));
      audio.play().catch(reject);
    });
  };

  const start = async () => {
    if (!a || !b) return;
    stop();
    setError('');
    const mine = token.current;
    try {
      await playOne('a', a, mine);
      if (mine === token.current) await playOne('b', b, mine);
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      if (mine === token.current) setPlaying(null);
    }
  };

  return (
    <div role="group" aria-label={t('qwenPlayground.ab_title')} className="flex flex-wrap items-center gap-2 rounded-xl border border-border/60 bg-card/30 p-2 text-xs">
      <span className="font-medium">{t('qwenPlayground.ab_title')}</span>
      <span className={playing === 'a' ? 'font-semibold text-primary' : ''}>A: {a?.label ?? t('qwenPlayground.ab_empty')}</span>
      <span className={playing === 'b' ? 'font-semibold text-primary' : ''}>B: {b?.label ?? t('qwenPlayground.ab_empty')}</span>
      {playing ? (
        <Button size="sm" variant="outline" onClick={stop}>
          <SquareIcon /> {t('qwenPlayground.ab_stop')}
        </Button>
      ) : (
        <Button size="sm" disabled={!a || !b} onClick={() => void start()}>
          <PlayIcon /> {t('qwenPlayground.ab_play')}
        </Button>
      )}
      <Button size="sm" variant="ghost" disabled={!a && !b} onClick={onClear}>
        {t('qwenPlayground.ab_clear')}
      </Button>
      {error && <span className="text-destructive">{error}</span>}
    </div>
  );
}
