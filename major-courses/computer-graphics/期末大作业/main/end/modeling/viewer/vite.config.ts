import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const viewerDir = path.dirname(fileURLToPath(import.meta.url));
const pipelineBuild = path.resolve(viewerDir, '../../../../build/modeling-pipeline');

export default defineConfig({
  publicDir: path.resolve(process.env.PIPELINE_PUBLIC_DIR || path.join(pipelineBuild, 'web-public')),
  build: {
    outDir: path.resolve(process.env.PIPELINE_WEB_OUT || path.join(pipelineBuild, 'web-dist')),
    // Output can be overridden by callers; never recursively delete their directory.
    emptyOutDir: false,
  },
});
