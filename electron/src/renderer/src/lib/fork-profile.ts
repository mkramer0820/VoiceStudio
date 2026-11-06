/**
 * What this fork shows. The fork is a Windows + CUDA story studio: Qwen3 playground and story, Fish Story and the training-data export,
 * transcripts, saved voices and the OpenAI-compatible API. The other workspaces of the upstream app stay in the code, and their routes and
 * API still work, but they are not offered in the menu, the command palette or the home page.
 *
 * To bring one back, delete its path from HIDDEN_ROUTES. This only decides what is *offered*: it never decides which model or engine
 * is used for anything.
 */
export const HIDDEN_ROUTES: ReadonlySet<string> = new Set([
  '/clone', // the upstream Clone page (Qwen clone/design live in the Qwen playground and Qwen story)
  '/design', // the upstream Voice design page
  '/stories', // the upstream multi-voice stories page (Qwen story and Fish Story replace it)
  '/audiobook', // the upstream audiobook builder
  '/dub', // video dubbing
  '/batch', // batch dubbing queue
  '/calls', // workflows / phone calls
  '/gallery', // community voice gallery
  '/projects', // dub and long-form project library
  '/tools', // upstream utility tools
]);

export const isRouteVisible = (path: string): boolean => !HIDDEN_ROUTES.has(path);
