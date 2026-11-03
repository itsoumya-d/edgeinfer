// One-off sync: wraps src/webgpu/kernels/paged_attention.wgsl as a TS string
// module so the SDK can inline the shader without a bundler plugin.
// The unit test asserts the two stay identical.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const wgsl = readFileSync(join(here, '../src/webgpu/kernels/paged_attention.wgsl'), 'utf8').replace(/\r\n/g, '\n');
const out = `// GENERATED from src/webgpu/kernels/paged_attention.wgsl — keep in sync.
// tests/paged-attention-wgsl.test.mjs asserts this string equals the .wgsl file.

/** WGSL source for the paged-attention decode kernel (float32, single layer). */
export const PAGED_ATTENTION_WGSL: string = /* wgsl */ \`${wgsl.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${')}\`;
`;
writeFileSync(join(here, '../src/webgpu/paged-attention-wgsl.ts'), out);
console.log('wrote src/webgpu/paged-attention-wgsl.ts');
