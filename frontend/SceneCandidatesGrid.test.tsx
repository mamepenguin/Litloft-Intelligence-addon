import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { NextIntlClientProvider } from 'next-intl';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import SceneCandidatesSection from '@/addons/intelligence/SceneCandidatesSection';
import { fetchSceneSegments } from '@/addons/intelligence/api';
import messages from '@/messages/en.json';

vi.unmock('next-intl');

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}));

vi.mock('@/addons/intelligence/api', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/addons/intelligence/api')>(),
  fetchSceneSegments: vi.fn(),
}));

const initialProps = {
  query: 'sunset',
  drive: 'home',
  includeSceneClip: true,
  typeFilter: null as string | null,
  smartFolderId: null as string | null,
};

function candidates(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    file_id: `scene-${index + 1}`,
    filename: `Rank ${index + 1}.mp4`,
    start: index * 10,
    end: index * 10 + 5,
    peak: index * 10 + 2,
    score: 1 - index / 100,
  }));
}

function section(props = initialProps) {
  return (
    <NextIntlClientProvider locale="en" messages={messages}>
      <SceneCandidatesSection {...props} />
    </NextIntlClientProvider>
  );
}

function visibleFilenames() {
  return screen.getAllByRole('link').map((link) =>
    link.textContent?.match(/Rank \d+\.mp4/)?.[0],
  ).filter(Boolean);
}

describe('scene candidate visibility (SPEC-ADDON-045)', () => {
  beforeEach(() => {
    vi.mocked(fetchSceneSegments).mockReset();
    vi.mocked(fetchSceneSegments).mockResolvedValue({
      query: initialProps.query,
      segments: candidates(20),
      partial: false,
    });
  });

  it('shows the first eight ranked candidates and expands and collapses without another search', async () => {
    render(section());

    const expand = await screen.findByRole('button', { name: /show all/i });
    expect(visibleFilenames()).toEqual(candidates(8).map(({ filename }) => filename));
    expect(expand.textContent).toContain('20');
    expect(expand).toHaveAttribute('aria-expanded', 'false');
    const requests = vi.mocked(fetchSceneSegments).mock.calls.length;

    fireEvent.click(expand);

    expect(visibleFilenames()).toEqual(candidates(20).map(({ filename }) => filename));
    const collapse = screen.getByRole('button', { name: /collapse/i });
    expect(collapse).toHaveAttribute('aria-expanded', 'true');
    expect(fetchSceneSegments).toHaveBeenCalledTimes(requests);

    fireEvent.click(collapse);

    expect(visibleFilenames()).toEqual(candidates(8).map(({ filename }) => filename));
    expect(screen.getByRole('button', { name: /show all/i })).toHaveAttribute('aria-expanded', 'false');
    expect(fetchSceneSegments).toHaveBeenCalledTimes(requests);
  });

  it.each([
    ['query', { query: 'ocean' }],
    ['drive', { drive: 'archive' }],
    ['scene search toggle', { includeSceneClip: false }],
    ['type filter', { typeFilter: 'video' }],
    ['smart folder', { smartFolderId: 'saved-search' }],
  ])('returns to the first eight when the %s changes', async (_name, change) => {
    const view = render(section());
    fireEvent.click(await screen.findByRole('button', { name: /show all/i }));
    expect(visibleFilenames()).toHaveLength(20);

    view.rerender(section({ ...initialProps, ...change }));
    await waitFor(() => expect(screen.queryByText('Rank 9.mp4')).not.toBeInTheDocument());
    view.rerender(section());

    const expand = await screen.findByRole('button', { name: /show all/i });
    expect(expand).toHaveAttribute('aria-expanded', 'false');
    expect(visibleFilenames()).toEqual(candidates(8).map(({ filename }) => filename));
  });

  it.each([1, 8])('shows all %i candidates without an expansion control', async (count) => {
    vi.mocked(fetchSceneSegments).mockResolvedValue({
      query: initialProps.query,
      segments: candidates(count),
      partial: false,
    });
    render(section());

    await screen.findByText('Rank 1.mp4');

    expect(visibleFilenames()).toEqual(candidates(count).map(({ filename }) => filename));
    expect(screen.queryByRole('button', { name: /show all|collapse/i })).not.toBeInTheDocument();
  });
});
