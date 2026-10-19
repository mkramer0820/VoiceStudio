import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn(), engines: vi.fn(), invalidate: vi.fn() }));

vi.mock('@/lib/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/client')>()),
  apiJson: mock.api,
}));
vi.mock('@/hooks/use-engines', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-engines')>()),
  useEngines: mock.engines,
}));
vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useQueryClient: () => ({ invalidateQueries: mock.invalidate }),
}));
vi.mock('@tanstack/react-router', () => ({
  Link: ({ children }: { children: React.ReactNode }) => <a href="#">{children}</a>,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

import { FishVoicePicker } from './fish-voice-picker';

const enginesWith = (fish?: { available: boolean }) => ({
  data: {
    tts: {
      active: 'omnivoice',
      backends: [
        { id: 'omnivoice', available: true },
        ...(fish ? [{ id: 'fish-audio', ...fish }] : []),
      ],
    },
  },
});

const VOICES = {
  voices: [
    { id: 'v1', title: 'Narrator', languages: ['en'], visibility: 'private' },
    { id: 'v2', title: 'Villain', languages: ['en'], visibility: 'private' },
  ],
  default_voice_id: 'v1',
  count: 2,
  scope: 'mine',
};

const KEY_NONE = { has_key: false, stored: false, from_env: false };

beforeEach(() => {
  mock.api.mockImplementation((path: string) =>
    path === '/api/fish/key' ? Promise.resolve(KEY_NONE) : Promise.resolve(VOICES),
  );
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it('renders nothing when the Fish engine is not registered', () => {
  mock.engines.mockReturnValue(enginesWith());
  const { container } = render(<FishVoicePicker />);
  expect(container).toBeEmptyDOMElement();
});

it('asks for a key and makes no request while the engine is unavailable', () => {
  mock.engines.mockReturnValue(enginesWith({ available: false }));
  render(<FishVoicePicker />);
  expect(screen.getByText('settings.fish_need_key')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'settings.fish_load' })).not.toBeInTheDocument();
  expect(mock.api).not.toHaveBeenCalledWith(expect.stringContaining('/api/fish/voices'));
});

it('contacts Fish only when Load is pressed, then marks the default voice', async () => {
  mock.engines.mockReturnValue(enginesWith({ available: true }));
  render(<FishVoicePicker />);
  expect(mock.api).not.toHaveBeenCalledWith(expect.stringContaining('/api/fish/voices'));

  fireEvent.click(screen.getByRole('button', { name: 'settings.fish_load' }));
  const rows = await screen.findAllByRole('radio', { name: /Narrator|Villain/ });
  expect(mock.api).toHaveBeenCalledWith('/api/fish/voices?scope=mine');
  expect(rows[0]).toHaveAttribute('aria-checked', 'true');
  expect(rows[1]).toHaveAttribute('aria-checked', 'false');
});

it('saves the chosen voice as the default', async () => {
  mock.engines.mockReturnValue(enginesWith({ available: true }));
  mock.api.mockImplementation((path: string) =>
    path === '/api/fish/key'
      ? Promise.resolve(KEY_NONE)
      : path.startsWith('/api/fish/voices')
        ? Promise.resolve(VOICES)
        : Promise.resolve({ default_voice_id: 'v2' }),
  );
  render(<FishVoicePicker />);
  fireEvent.click(screen.getByRole('button', { name: 'settings.fish_load' }));
  fireEvent.click(await screen.findByRole('radio', { name: /Villain/ }));

  await waitFor(() =>
    expect(mock.api).toHaveBeenCalledWith('/api/fish/default-voice', {
      method: 'PUT',
      body: JSON.stringify({ voice_id: 'v2' }),
    }),
  );
  await waitFor(() =>
    expect(screen.getByRole('radio', { name: /Villain/ })).toHaveAttribute('aria-checked', 'true'),
  );
});

it('searches the public library with the typed title', async () => {
  mock.engines.mockReturnValue(enginesWith({ available: true }));
  mock.api.mockImplementation((path: string) =>
    Promise.resolve(path === '/api/fish/key' ? KEY_NONE : { ...VOICES, voices: [], scope: 'library' }),
  );
  render(<FishVoicePicker />);
  fireEvent.click(screen.getByRole('radio', { name: 'settings.fish_scope_library' }));
  fireEvent.change(screen.getByLabelText('settings.fish_search'), { target: { value: 'calm' } });
  fireEvent.click(screen.getByRole('button', { name: 'settings.fish_load' }));

  await waitFor(() =>
    expect(mock.api).toHaveBeenCalledWith('/api/fish/voices?scope=library&title=calm'),
  );
  expect(await screen.findByText('settings.fish_empty')).toBeInTheDocument();
});

it('shows the Fish error message when listing fails', async () => {
  mock.engines.mockReturnValue(enginesWith({ available: true }));
  mock.api.mockImplementation((path: string) =>
    path === '/api/fish/key'
      ? Promise.resolve(KEY_NONE)
      : Promise.reject(new Error('Fish Audio rejected the API key')),
  );
  render(<FishVoicePicker />);
  fireEvent.click(screen.getByRole('button', { name: 'settings.fish_load' }));
  expect(await screen.findByRole('alert')).toBeInTheDocument();
});

it('stores the key through the backend and never echoes it back', async () => {
  mock.engines.mockReturnValue(enginesWith({ available: false }));
  mock.api.mockImplementation((_path: string, init?: RequestInit) =>
    Promise.resolve(init?.method === 'PUT' ? { has_key: true, stored: true, from_env: false } : KEY_NONE),
  );
  render(<FishVoicePicker />);
  const input = screen.getByPlaceholderText('settings.llmp_key_paste') as HTMLInputElement;
  expect(input.type).toBe('password');
  fireEvent.change(input, { target: { value: 'fk-secret-123456' } });
  fireEvent.click(screen.getByRole('button', { name: 'common.save' }));

  await waitFor(() =>
    expect(mock.api).toHaveBeenCalledWith('/api/fish/key', {
      method: 'PUT',
      body: JSON.stringify({ api_key: 'fk-secret-123456' }),
    }),
  );
  await waitFor(() => expect(mock.invalidate).toHaveBeenCalled());
  await waitFor(() => expect(input.value).toBe(''));
  expect(screen.getByRole('button', { name: 'settings.fish_key_remove' })).toBeInTheDocument();
});
