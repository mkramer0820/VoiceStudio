import { CheckCircle2Icon, CloudIcon, LoaderCircleIcon, SearchIcon } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useEngines, engineFamilyState } from '@/hooks/use-engines';
import { apiJson, describeError } from '@/lib/api/client';
import { queryKeys } from '@/lib/query';
import { cn } from '@/lib/utils';
import { SettingsActionError } from './settings-action-error';
import { SettingsRow, SettingsSection } from './settings-layout';

export const FISH_ENGINE_ID = 'fish-audio';

type Scope = 'mine' | 'library';

export interface FishVoice {
  id: string;
  title: string;
  description?: string;
  languages?: string[];
  tags?: string[];
  visibility?: string | null;
  state?: string | null;
  author?: string | null;
}

interface FishKeyStatus {
  has_key: boolean;
  stored: boolean;
  from_env: boolean;
}

interface FishVoicesResponse {
  voices: FishVoice[];
  default_voice_id: string | null;
  count: number;
  scope: Scope;
}

/**
 * Pick the Fish Audio voice used when a request names none. Loading the list
 * contacts Fish, so it only happens when the user presses the button.
 */
export function FishVoicePicker() {
  const { t } = useTranslation();
  const engines = useEngines();
  const queryClient = useQueryClient();
  const fish = engineFamilyState(engines.data, 'tts')?.backends.find(
    (backend) => backend.id === FISH_ENGINE_ID,
  );
  const [scope, setScope] = useState<Scope>('mine');
  const [query, setQuery] = useState('');
  const [voices, setVoices] = useState<FishVoice[] | null>(null);
  const [defaultId, setDefaultId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [keyStatus, setKeyStatus] = useState<FishKeyStatus | null>(null);
  const [keyInput, setKeyInput] = useState('');
  const [keySaving, setKeySaving] = useState(false);
  const registered = Boolean(fish);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!registered) return;
    apiJson<FishKeyStatus>('/api/fish/key')
      .then(setKeyStatus)
      .catch((reason) => setError(describeError(reason)));
  }, [registered]);

  if (!fish) return null;

  const saveKey = async (remove: boolean) => {
    if (keySaving) return;
    setKeySaving(true);
    setError('');
    try {
      const next = await apiJson<FishKeyStatus>('/api/fish/key', {
        method: remove ? 'DELETE' : 'PUT',
        ...(remove ? {} : { body: JSON.stringify({ api_key: keyInput.trim() }) }),
      });
      setKeyStatus(next);
      setKeyInput('');
      if (remove) setVoices(null);
      await queryClient.invalidateQueries({ queryKey: queryKeys.engines });
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setKeySaving(false);
    }
  };

  const load = async (nextScope: Scope = scope) => {
    if (loading) return;
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({ scope: nextScope });
      if (query.trim()) params.set('title', query.trim());
      const result = await apiJson<FishVoicesResponse>(`/api/fish/voices?${params}`);
      setVoices(result.voices);
      setDefaultId(result.default_voice_id);
      // The list renders below the controls; bring it into view so a load is never silent.
      requestAnimationFrame(() => listRef.current?.scrollIntoView?.({ block: 'nearest' }));
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setLoading(false);
    }
  };

  const choose = async (voiceId: string | null) => {
    if (saving) return;
    setSaving(voiceId ?? '');
    setError('');
    try {
      const result = await apiJson<{ default_voice_id: string | null }>(
        '/api/fish/default-voice',
        { method: 'PUT', body: JSON.stringify({ voice_id: voiceId }) },
      );
      setDefaultId(result.default_voice_id);
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setSaving(null);
    }
  };

  const changeScope = (next: Scope) => {
    setScope(next);
    setVoices(null);
  };

  return (
    <SettingsSection icon={CloudIcon} title={t('settings.fish_title')}>
      <SettingsRow
        id="fish-voices-overview"
        title={t('settings.fish_desc')}
        description={fish.available ? t('settings.fish_privacy') : t('settings.fish_need_key')}
      >
        {null}
      </SettingsRow>

      {error && (
        <div className="p-4">
          <SettingsActionError
            title={t('settings.fish_error')}
            detail={error}
            onDismiss={() => setError('')}
          />
        </div>
      )}

      <SettingsRow
        id="fish-key"
        title={t('settings.fish_key_title')}
        description={
          keyStatus?.from_env
            ? t('settings.fish_key_env')
            : keyStatus?.stored
              ? t('settings.llmp_key_stored')
              : t('settings.llmp_key_paste')
        }
      >
        <Input
          aria-label={t('settings.fish_key_title')}
          type="password"
          value={keyInput}
          className="w-full font-mono @2xl:w-72"
          placeholder={keyStatus?.stored ? '••••••••' : t('settings.llmp_key_paste')}
          autoComplete="off"
          disabled={keySaving}
          onChange={(event) => setKeyInput(event.target.value)}
        />
        <Button disabled={keySaving || keyInput.trim().length < 8} onClick={() => void saveKey(false)}>
          {keySaving && <LoaderCircleIcon className="animate-spin" />}
          {keySaving ? t('common.saving') : t('common.save')}
        </Button>
        {keyStatus?.stored && (
          <Button variant="outline" disabled={keySaving} onClick={() => void saveKey(true)}>
            {t('settings.fish_key_remove')}
          </Button>
        )}
      </SettingsRow>

      {fish.available && (
        <>
          <SettingsRow
            id="fish-voices-load"
            title={t('settings.fish_load_title')}
            description={voices ? t('settings.fish_loaded', { count: voices.length }) : undefined}
          >
            <div role="radiogroup" aria-label={t('settings.fish_scope')} className="flex gap-1">
              {(['mine', 'library'] as const).map((item) => (
                <Button
                  key={item}
                  role="radio"
                  aria-checked={scope === item}
                  variant={scope === item ? 'secondary' : 'outline'}
                  size="sm"
                  disabled={loading}
                  onClick={() => changeScope(item)}
                >
                  {t(item === 'mine' ? 'settings.fish_scope_mine' : 'settings.fish_scope_library')}
                </Button>
              ))}
            </div>
            <Input
              aria-label={t('settings.fish_search')}
              value={query}
              placeholder={t('settings.fish_search')}
              className="w-full @2xl:w-56"
              disabled={loading}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') void load();
              }}
            />
            <Button disabled={loading} onClick={() => void load()}>
              {loading ? <LoaderCircleIcon className="animate-spin" /> : <SearchIcon />}
              {loading ? t('common.loading') : t('settings.fish_load')}
            </Button>
          </SettingsRow>

          <SettingsRow
            id="fish-voices-default"
            title={t('settings.fish_default')}
            description={defaultId ?? t('settings.fish_default_none')}
          >
            {defaultId && (
              <Button variant="outline" disabled={saving !== null} onClick={() => void choose(null)}>
                {t('settings.fish_clear')}
              </Button>
            )}
          </SettingsRow>

          {voices && (
            <div
              ref={listRef}
              role="radiogroup"
              aria-label={t('settings.fish_voices')}
              className="max-h-96 space-y-1 overflow-y-auto p-2"
            >
              {voices.length === 0 && (
                <p className="p-3 text-sm text-muted-foreground">{t('settings.fish_empty')}</p>
              )}
              {voices.map((voice) => {
                const selected = voice.id === defaultId;
                return (
                  <button
                    key={voice.id}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    disabled={saving !== null}
                    onClick={() => void choose(voice.id)}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none',
                      selected && 'bg-accent/60',
                    )}
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-medium">{voice.title || voice.id}</span>
                      <span className="block truncate text-xs text-muted-foreground">
                        {[voice.languages?.join(', '), voice.tags?.join(', '), voice.author]
                          .filter(Boolean)
                          .join(' · ')}
                      </span>
                    </span>
                    {voice.visibility && (
                      <span className="shrink-0 text-xs text-muted-foreground">
                        {voice.visibility}
                      </span>
                    )}
                    {saving === voice.id ? (
                      <LoaderCircleIcon className="size-4 shrink-0 animate-spin" />
                    ) : (
                      selected && (
                        <CheckCircle2Icon
                          aria-label={t('settings.fish_selected')}
                          className="size-4 shrink-0 text-emerald-500"
                        />
                      )
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </>
      )}
    </SettingsSection>
  );
}
