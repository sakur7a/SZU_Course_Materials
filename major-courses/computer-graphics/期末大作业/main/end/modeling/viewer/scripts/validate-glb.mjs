import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import validator from 'gltf-validator';

const [input, output, ...extra] = process.argv.slice(2);
if (!input || extra.length) {
  console.error('Usage: node scripts/validate-glb.mjs <input.glb> [report.json]');
  process.exit(2);
}

try {
  const source = path.resolve(input);
  const bytes = await readFile(source);
  const report = await validator.validateBytes(new Uint8Array(bytes), {
    uri: path.basename(source),
    format: 'glb',
    maxIssues: 0,
    writeTimestamp: false,
  });
  const json = `${JSON.stringify(report, null, 2)}\n`;
  if (output) {
    await writeFile(path.resolve(output), json, 'utf8');
    console.log(`glTF Validator: ${report.issues.numErrors} errors, ${report.issues.numWarnings} warnings; report: ${output}`);
  } else {
    process.stdout.write(json);
  }
  if (report.issues.numErrors > 0) process.exitCode = 1;
} catch (error) {
  console.error(`glTF validation failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 2;
}
