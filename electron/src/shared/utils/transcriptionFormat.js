export function segTimeRange(seg) {
  const known = (v) => typeof v === 'number' && Number.isFinite(v);
  const start = known(seg?.start) ? `${seg.start.toFixed(1)}s` : null;
  const end = known(seg?.end) ? `${seg.end.toFixed(1)}s` : null;
  if (start && end) return `${start} – ${end}`;
  return start || end || '';
}

/** Cleaned dictation text is the user-facing result; raw ASR stays recoverable. */
export function preferredTranscript(entry) {
  return entry?.refined_text?.trim() || entry?.text || '';
}

/** Keep language and recording date attached when exporting history. */
export function formatTranscriptExport(entries) {
  return entries
    .map((entry) => {
      const date = new Date(entry.timestamp);
      const stamp = Number.isFinite(date.getTime()) ? date.toLocaleString() : '';
      const text = preferredTranscript(entry);
      return `[${stamp}] (${entry.language || ''})\n${text}\n`;
    })
    .join('\n---\n\n');
}


const timed = (seg) => typeof seg?.start === 'number' && Number.isFinite(seg.start);

/** Segments that carry a start time. A backend answering in plain text format has none, and nothing here invents them. */
export function timedSegments(entry) {
  return (entry?.segments || []).filter((seg) => timed(seg) && String(seg.text || '').trim());
}

/** Segment end for a subtitle cue: its own, else the next cue's start, else a short reading time (the ASR gave none). */
function cueEnd(segs, i) {
  const seg = segs[i];
  if (typeof seg.end === 'number' && Number.isFinite(seg.end) && seg.end > seg.start) return seg.end;
  const next = segs[i + 1];
  if (next && next.start > seg.start) return next.start;
  return seg.start + Math.max(1, String(seg.text).length / 15);
}

function clock(seconds, comma) {
  const ms = Math.max(0, Math.round(seconds * 1000));
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const sec = Math.floor((ms % 60000) / 1000);
  const rest = ms % 1000;
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${pad(h)}:${pad(m)}:${pad(sec)}${comma ? ',' : '.'}${pad(rest, 3)}`;
}

/** SubRip (.srt) from the timed segments of one transcript. Empty string when none are timed. */
export function formatSrt(entry) {
  const segs = timedSegments(entry);
  return segs
    .map((seg, i) => `${i + 1}\n${clock(seg.start, true)} --> ${clock(cueEnd(segs, i), true)}\n${String(seg.text).trim()}\n`)
    .join('\n');
}

/** WebVTT (.vtt) from the timed segments of one transcript. Empty string when none are timed. */
export function formatVtt(entry) {
  const segs = timedSegments(entry);
  if (!segs.length) return '';
  const cues = segs.map((seg, i) => `${clock(seg.start, false)} --> ${clock(cueEnd(segs, i), false)}\n${String(seg.text).trim()}\n`);
  return `WEBVTT\n\n${cues.join('\n')}`;
}

/** Plain text with a [mm:ss] stamp in front of each timed segment. */
export function formatTimedText(entry) {
  return timedSegments(entry)
    .map((seg) => `[${clock(seg.start, false).slice(0, 8)}] ${String(seg.text).trim()}`)
    .join('\n');
}

/** Everything the transcript knows, for other tools: text, language, and every segment with its start and end as the ASR gave them. */
export function formatSegmentsJson(entry) {
  return JSON.stringify(
    {
      text: preferredTranscript(entry),
      language: entry?.language || '',
      duration_s: entry?.duration_s ?? null,
      recorded: entry?.timestamp || '',
      segments: (entry?.segments || []).map((seg) => ({
        start: typeof seg.start === 'number' ? seg.start : null,
        end: typeof seg.end === 'number' ? seg.end : null,
        text: seg.text,
      })),
    },
    null,
    2,
  );
}
