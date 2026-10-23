import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, describeError } from '@/lib/api/client';

/** One clip at a time; audio is fetched through the API client so authentication applies. */
export function useClipPlayer(onError: (message: string) => void) {
  const [playing, setPlaying] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const url = useRef<string | null>(null);

  const stop = useCallback(() => {
    audio.current?.pause();
    audio.current = null;
    if (url.current) URL.revokeObjectURL(url.current);
    url.current = null;
    setPlaying(null);
  }, []);

  useEffect(() => stop, [stop]);

  const toggle = useCallback(
    async (key: string) => {
      if (playing === key) return stop();
      stop();
      try {
        const res = await apiFetch(`/api/fish/dataset/audio?key=${encodeURIComponent(key)}`);
        url.current = URL.createObjectURL(await res.blob());
        const next = new Audio(url.current);
        next.onended = stop;
        audio.current = next;
        setPlaying(key);
        await next.play();
      } catch (reason) {
        stop();
        onError(describeError(reason));
      }
    },
    [playing, stop, onError],
  );

  return { playing, toggle };
}
