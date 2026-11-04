import type { TranscriptEntry, TranscriptSegment } from './transcriptionsStore';
export function segTimeRange(segment?: TranscriptSegment | null): string;
export function preferredTranscript(entry: TranscriptEntry): string;
export function formatTranscriptExport(entries: TranscriptEntry[]): string;
export function timedSegments(entry: TranscriptEntry): TranscriptSegment[];
export function formatSrt(entry: TranscriptEntry): string;
export function formatVtt(entry: TranscriptEntry): string;
export function formatTimedText(entry: TranscriptEntry): string;
export function formatSegmentsJson(entry: TranscriptEntry): string;
