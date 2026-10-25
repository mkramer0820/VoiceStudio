import {
  ChevronDownIcon,
  ChevronRightIcon,
  FolderOpenIcon,
  FolderPlusIcon,
  LayersIcon,
} from 'lucide-react';
import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { getBridge } from '@/components/bridge';
import { Button } from '@/components/ui/button';
import { apiJson, describeError } from '@/lib/api/client';
import { SettingsRow, SettingsSection } from './settings-layout';

interface LocationModel {
  id: string;
  label: string;
  path: string;
  type: 'custom_voice' | 'base' | 'voice_design';
  size?: string;
  speakers: string[];
}

interface ModelLocation {
  id: string;
  name: string;
  path: string;
  exists: boolean;
  error: string | null;
  models_found?: number;
  models: LocationModel[];
}

interface ModelsDirectoryState {
  configured: string | null;
  effective?: string;
  default: string;
}

const queryKey = ['model-locations'] as const;
const endpoint = '/api/settings/storage/model-locations';

function LocationRow({
  row,
  busy,
  onChange,
  onRemove,
}: {
  row: ModelLocation;
  busy: boolean;
  onChange?: () => void;
  onRemove: () => void;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const found = row.models_found ?? row.models.length;
  const status = !row.exists
    ? t('modelLocations.status_missing')
    : row.error
      ? t('modelLocations.status_error', { error: row.error })
      : found === 0
        ? t('modelLocations.status_empty')
        : t('modelLocations.status_found', { count: found });
  const problem = !row.exists || !!row.error || found === 0;
  return (
    <SettingsRow
      id={`model-location-${row.id}`}
      title={row.name}
      description={
        <>
          <code title={row.path} className="block break-all font-mono text-xs">
            {row.path}
          </code>
          <span role="status" className={problem ? 'text-amber-500' : undefined}>
            {status}
          </span>
        </>
      }
    >
      {onChange && (
        <Button size="sm" variant="outline" disabled={busy} onClick={onChange}>
          <FolderOpenIcon />
          {t('modelLocations.change')}
        </Button>
      )}
      <Button size="sm" variant="ghost" disabled={busy} onClick={onRemove}>
        {t('modelLocations.remove')}
      </Button>
      {found > 0 && (
        <Button size="sm" variant="ghost" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          {open ? <ChevronDownIcon /> : <ChevronRightIcon />}
          {t('modelLocations.models_here')}
        </Button>
      )}
      {open && (
        <ul className="w-full space-y-2 pt-1 text-xs">
          {row.models.map((m) => (
            <li key={m.id} className="rounded-md border border-border/50 bg-muted/25 p-2">
              <div className="font-medium">{m.label}</div>
              <div className="text-muted-foreground">
                {t('modelLocations.model_line', {
                  type: t(`modelLocations.type_${m.type}`),
                  size: m.size ?? '',
                })}
                {' · '}
                {m.speakers.length
                  ? t('modelLocations.speakers', { list: m.speakers.join(', ') })
                  : t('modelLocations.no_speakers')}
              </div>
              <code className="block break-all font-mono text-muted-foreground">{m.path}</code>
            </li>
          ))}
        </ul>
      )}
    </SettingsRow>
  );
}

export function ModelLocationsSettings() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const bridge = getBridge();
  const query = useQuery({
    queryKey,
    queryFn: ({ signal }) => apiJson<{ locations: ModelLocation[] }>(endpoint, { signal }),
  });
  const cache = useQuery({
    queryKey: ['models-directory'],
    queryFn: ({ signal }) =>
      apiJson<ModelsDirectoryState>('/api/settings/storage/models-dir', { signal }),
  });

  const run = async (work: () => Promise<void>) => {
    if (busy) return;
    setBusy(true);
    setError('');
    try {
      await work();
      await client.invalidateQueries({ queryKey });
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };

  // The renderer only ever sends the opaque authorization; the folder path never travels from here.
  const pick = (existing?: ModelLocation) =>
    run(async () => {
      const selected = await bridge?.files.authorizeModelLocation?.();
      if (!selected) return;
      if (existing) {
        await apiJson(`${endpoint}/${encodeURIComponent(existing.id)}`, {
          method: 'PUT',
          body: JSON.stringify({ authorization: selected.authorization }),
        });
      } else {
        await apiJson(endpoint, {
          method: 'POST',
          body: JSON.stringify({ authorization: selected.authorization }),
        });
      }
    });
  const remove = (row: ModelLocation) =>
    run(async () => {
      await apiJson(`${endpoint}/${encodeURIComponent(row.id)}`, { method: 'DELETE' });
    });

  const canPick = !!bridge?.files.authorizeModelLocation;
  const rows = query.data?.locations ?? [];
  const cachePath = cache.data?.configured || cache.data?.effective || cache.data?.default;
  return (
    <SettingsSection icon={LayersIcon} title={t('modelLocations.title')}>
      <SettingsRow
        id="model-locations-hf"
        title={t('modelLocations.hf_title')}
        description={t('modelLocations.hf_desc')}
      >
        <code
          title={cachePath}
          className="min-w-0 max-w-full flex-1 truncate rounded-md border border-border/60 bg-muted/35 px-2.5 py-1.5 text-xs text-muted-foreground @2xl:max-w-[32rem]"
        >
          {cache.isPending ? t('common.loading') : cachePath}
        </code>
        <Button
          size="sm"
          variant="ghost"
          onClick={() =>
            document.getElementById('models-directory')?.scrollIntoView({ behavior: 'smooth' })
          }
        >
          {t('modelLocations.hf_change')}
        </Button>
      </SettingsRow>
      {rows.map((row) => (
        <LocationRow
          key={row.id}
          row={row}
          busy={busy}
          onChange={canPick ? () => void pick(row) : undefined}
          onRemove={() => void remove(row)}
        />
      ))}
      <SettingsRow
        id="model-locations-add"
        title={t('modelLocations.add_title')}
        description={rows.length === 0 ? t('modelLocations.empty') : t('modelLocations.remove_note')}
      >
        {canPick && (
          <Button
            size="sm"
            variant="outline"
            disabled={busy || query.isPending}
            onClick={() => void pick()}
          >
            <FolderPlusIcon />
            {t('modelLocations.add')}
          </Button>
        )}
      </SettingsRow>
      {(query.isError || error) && (
        <div role="alert" className="flex items-center gap-2 px-4 py-3 text-sm text-destructive">
          <span className="min-w-0 flex-1 break-words">{error || t('common.error')}</span>
          {query.isError && (
            <Button size="sm" variant="ghost" onClick={() => void query.refetch()}>
              {t('common.retry')}
            </Button>
          )}
        </div>
      )}
    </SettingsSection>
  );
}
