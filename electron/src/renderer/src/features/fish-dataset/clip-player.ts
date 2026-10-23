import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, describeError } from '@/lib/api/client';

/**
 * One clip at a time; audio is fetched through the API client so authentication applies.
 * ``onEnded`` fires when a clip finishes by itself (not when you stop it), which drives auto-advance.
 */
export function useClipPlayer(onError: (message: string) => void, onEnded?: (key: string) => void) {
  const [playing, setPlaying] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const url = useRef<string | null>(null);
  const ended = useRef(onEnded);
  ended.current = onEnded;
  const token = useRef(0);

  const stop = useCallback(() => {
    token.current += 1;
    audio.current?.pause();
    audio.current = null;
    if (url.current) URL.revokeObjectURL(url.current);
    url.current = null;
    setPlaying(null);
  }, []);

  useEffect(() => stop, [stop]);

  const play = useCallback(
    async (key: string) => {
      stop();
      const mine = token.current;
      try {
        const res = await apiFetch(`/api/fish/dataset/audio?key=${encodeURIComponent(key)}`);
        const blob = await res.blob();
        if (mine !== token.current) return; // another clip was chosen while this one loaded
        url.current = URL.createObjectURL(blob);
        const next = new Audio(url.current);
        next.onended = () => {
          if (mine !== token.current) return;
          stop();
          ended.current?.(key);
        };
        audio.current = next;
        setPlaying(key);
        await next.play();
      } catch (reason) {
        if (mine !== token.current) return;
        stop();
        onError(describeError(reason));
      }
    },
    [stop, onError],
  );

  const toggle = useCallback(
    async (key: string) => {
      if (playing === key) return stop();
      await play(key);
    },
    [playing, play, stop],
  );

  return { playing, play, toggle, stop };
}
