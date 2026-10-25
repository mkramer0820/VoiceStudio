import { mkdir, mkdtemp, readdir, readFile, realpath, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, expect, it } from 'vitest';
import { authorizeModelLocation, authorizeModelsDirectory } from './media-authorization';

const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

it('mints one-shot models-directory capabilities only after a writable directory check', async () => {
  const root = await mkdtemp(join(tmpdir(), 'voicestudio-model-dir-'));
  roots.push(root);
  const dataDir = join(root, 'data');
  const selected = join(root, 'model cache');

  const result = await authorizeModelsDirectory(dataDir, selected);
  expect(result.path).toBe(await realpath(selected));
  const payload = JSON.parse(
    await readFile(join(dataDir, '.path-authorizations', `${result.authorization}.json`), 'utf8'),
  );
  expect(payload).toEqual({
    token: result.authorization,
    kind: 'models_dir',
    path: await realpath(selected),
  });
});

it('represents restoring the platform default with an authorized empty path', async () => {
  const root = await mkdtemp(join(tmpdir(), 'voicestudio-model-reset-'));
  roots.push(root);

  const result = await authorizeModelsDirectory(join(root, 'data'), '');
  const payload = JSON.parse(
    await readFile(
      join(root, 'data', '.path-authorizations', `${result.authorization}.json`),
      'utf8',
    ),
  );
  expect(result.path).toBe('');
  expect(payload.path).toBe('');
});

it('mints a model-location capability for an existing folder without writing into it', async () => {
  const root = await mkdtemp(join(tmpdir(), 'voicestudio-model-loc-'));
  roots.push(root);
  const folder = join(root, 'fine tunes');
  await mkdir(folder);

  const result = await authorizeModelLocation(join(root, 'data'), folder);
  const payload = JSON.parse(
    await readFile(
      join(root, 'data', '.path-authorizations', `${result.authorization}.json`),
      'utf8',
    ),
  );
  expect(payload).toEqual({
    token: result.authorization,
    kind: 'model_location',
    path: await realpath(folder),
  });
  expect(await readdir(folder)).toEqual([]);
  await expect(authorizeModelLocation(join(root, 'data'), join(root, 'missing'))).rejects.toThrow();
  await expect(authorizeModelLocation(join(root, 'data'), 'relative')).rejects.toThrow(
    'absolute',
  );
});
