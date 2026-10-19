import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn(), engines: vi.fn() }));

vi.mock('@/lib/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/client')>()),
  apiJson: mock.api,
}));
vi.mock('@/hooks/use-engines', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-engines')>()),
  useEngines: mock.engines,
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
  expect(mock.api).not.toHaveBeenCalled();
});

it('contacts Fish only when Load is pressed, then marks the default voice', async () => {
  mock.engines.mockReturnValue(enginesWith({ available: true }));
  mock.api.mockResolvedValue(VOICES);
  render(<FishVoicePicker />);
  expect(mock.api).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'settings.fish_load' }));
  const rows = await screen.findAllByRole('radio', { name: /Narrator|Villain/ });
  expect(mock.api).toHaveBeenCalledWith('/api/fish/voices?scope=mine');
  expect(rows[0]).toHaveAttribute('aria-checked', 'true');
  expect(rows[1]).toHaveAttribute('aria-checked', 'false');
});

it('saves the chosen voice as the default', async () => {
  mock.engines.mockReturnValue(enginesWith({ available: true }));
  mock.api.mockImplementation((path: string) =>
    path.startsWith('/api/fish/voices')
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
  mock.api.mockResolvedValue({ ...VOICES, voices: [], scope: 'library' });
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
  mock.api.mockRejectedValue(new Error('Fish Audio rejected the API key'));
  render(<FishVoicePicker />);
  fireEvent.click(screen.getByRole('button', { name: 'settings.fish_load' }));
  expect(await screen.findByRole('alert')).toBeInTheDocument();
});
