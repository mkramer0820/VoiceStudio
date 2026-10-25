import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn(), authorize: vi.fn() }));

vi.mock('@/lib/api/client', async (load) => {
  const actual = await load<typeof import('@/lib/api/client')>();
  return { ...actual, apiJson: mock.api };
});
vi.mock('@/components/bridge', () => ({
  getBridge: () => ({ files: { authorizeModelLocation: mock.authorize } }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, v?: Record<string, unknown>) => (v ? `${key} ${JSON.stringify(v)}` : key),
  }),
}));

import { ModelLocationsSettings } from './model-locations-settings';

const ROWS = [
  {
    id: 'tunes',
    name: 'My tunes',
    path: 'C:\\tunes',
    exists: true,
    error: null,
    models_found: 1,
    models: [
      {
        id: 'ck',
        label: 'ss_narrator',
        path: 'C:\\tunes\\ss_narrator',
        type: 'custom_voice',
        size: '1b7',
        speakers: ['ss_narrator'],
      },
    ],
  },
  { id: 'gone', name: 'Old', path: 'E:\\old', exists: false, error: 'folder missing', models: [] },
];

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderSettings() {
  mock.api.mockImplementation((path: string) => {
    if (path === '/api/settings/storage/models-dir') {
      return Promise.resolve({ configured: null, effective: 'C:\\hf', default: 'C:\\hf' });
    }
    if (path === '/api/settings/storage/model-locations') {
      return Promise.resolve({ locations: ROWS });
    }
    return Promise.resolve({});
  });
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <ModelLocationsSettings />
    </QueryClientProvider>,
  );
}

const writes = () => mock.api.mock.calls.filter(([, init]) => init?.method);

it('shows the HF cache, each folder with its status, and the models found inside', async () => {
  renderSettings();
  await screen.findByText('My tunes');
  expect(await screen.findByText('C:\\hf')).toBeTruthy();
  expect(screen.getByText('C:\\tunes')).toBeTruthy();
  expect(screen.getByText(/modelLocations\.status_found.*"count":1/)).toBeTruthy();
  expect(screen.getByText('modelLocations.status_missing')).toBeTruthy();
  expect(screen.queryByText('ss_narrator')).toBeNull();
  fireEvent.click(screen.getAllByText('modelLocations.models_here')[0]);
  expect(screen.getByText('ss_narrator')).toBeTruthy();
  expect(screen.getByText(/modelLocations\.model_line.*type_custom_voice/)).toBeTruthy();
  expect(screen.getByText('C:\\tunes\\ss_narrator')).toBeTruthy();
});

it('adds a folder by sending only the native authorization', async () => {
  mock.authorize.mockResolvedValue({ authorization: 'tok', path: 'D:\\secret path' });
  renderSettings();
  await screen.findByText('My tunes');
  fireEvent.click(screen.getByText('modelLocations.add'));
  await waitFor(() => expect(writes()).toHaveLength(1));
  const [url, init] = writes()[0];
  expect(url).toBe('/api/settings/storage/model-locations');
  expect(init.method).toBe('POST');
  expect(JSON.parse(init.body)).toEqual({ authorization: 'tok' });
  expect(init.body).not.toContain('secret');
});

it('does not write after the picker is cancelled', async () => {
  mock.authorize.mockResolvedValue(null);
  renderSettings();
  await screen.findByText('My tunes');
  fireEvent.click(screen.getByText('modelLocations.add'));
  await waitFor(() => expect(mock.authorize).toHaveBeenCalled());
  expect(writes()).toHaveLength(0);
});

it('changes a folder with PUT and removes one with DELETE', async () => {
  mock.authorize.mockResolvedValue({ authorization: 'tok2', path: 'D:\\x' });
  renderSettings();
  await screen.findByText('My tunes');
  fireEvent.click(screen.getAllByText('modelLocations.change')[0]);
  await waitFor(() => expect(writes()).toHaveLength(1));
  expect(writes()[0][0]).toBe('/api/settings/storage/model-locations/tunes');
  expect(writes()[0][1].method).toBe('PUT');
  fireEvent.click(screen.getAllByText('modelLocations.remove')[1]);
  await waitFor(() => expect(writes()).toHaveLength(2));
  expect(writes()[1][0]).toBe('/api/settings/storage/model-locations/gone');
  expect(writes()[1][1].method).toBe('DELETE');
});
