import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

vi.mock('@tanstack/react-router', () => ({
  useRouterState: ({ select }: any) => select({ location: { pathname: '/qwen-story' } }),
  Link: ({ to, activeProps: _activeProps, children, ...props }: any) => (
    <a href={to} {...props}>
      {children}
    </a>
  ),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@/lib/store/workspace', () => ({ setWorkspace: vi.fn() }));
import { HIDDEN_ROUTES, isRouteVisible } from '@/lib/fork-profile';
import { WorkspaceNavigation } from './workspace-menu';

it('offers only the studios and tools this fork uses', () => {
  render(<WorkspaceNavigation />);
  const hrefs = Array.from(document.querySelectorAll('nav a'), (a) => a.getAttribute('href'));
  for (const kept of ['/qwen-playground', '/qwen-story', '/fish-story', '/fish-dataset', '/transcriptions', '/personas', '/integrations']) {
    expect(hrefs).toContain(kept);
  }
  for (const hidden of HIDDEN_ROUTES) expect(hrefs).not.toContain(hidden);
  expect(screen.queryByRole('button', { name: 'nav.dub' })).toBeNull(); // the empty Dubbing group is not drawn
});

it('hides workspaces only from what is offered, never from the routes', () => {
  expect(isRouteVisible('/dub')).toBe(false);
  expect(isRouteVisible('/qwen-playground')).toBe(true);
  expect(isRouteVisible('/fish-dataset')).toBe(true);
});
