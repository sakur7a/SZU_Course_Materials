"""Regression checks for M0 export integrity and immutable acceptance evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
from asset_checks import check_package, sha256

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT/'build/modeling-pipeline'
POINTER = json.loads((BUILD/'pipeline-fixture/validated.json').read_text(encoding='utf-8'))
SOURCE = BUILD/'pipeline-fixture'/POINTER['run']


class PipelineIntegrity(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(prefix='negative-', dir=BUILD)).resolve()
        self.package = self.scratch/'asset'
        shutil.copytree(SOURCE, self.package)

    def tearDown(self):
        # Delete only the verified task-owned temporary directory.
        if self.scratch.parent != BUILD.resolve():
            raise RuntimeError('Refusing cleanup outside the pipeline build directory')
        shutil.rmtree(self.scratch)

    def check(self):
        return check_package(self.package, ROOT/'main/end/modeling/schemas')

    def test_real_package(self):
        self.assertEqual(self.check()['status'], 'passed')

    def test_legacy_transform_drift_even_with_updated_hash(self):
        path = self.package/'legacy/nodes.tsv'
        lines = path.read_text(encoding='utf-8').splitlines()
        row = lines[0].split('\t')
        row[16] = str(float(row[16])+0.25)
        lines[0] = '\t'.join(row)
        path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        manifest_path = self.package/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['files']['legacy/nodes.tsv'] = sha256(path)
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(ValueError): self.check()

    def test_modified_glb_is_rejected(self):
        path = self.package/'model.glb'
        content = bytearray(path.read_bytes()); content[-1] ^= 1; path.write_bytes(content)
        with self.assertRaises(ValueError): self.check()

    def test_escaping_resource_is_rejected(self):
        path = self.package/'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['files']['../outside.txt'] = '0'*64
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(ValueError): self.check()

    def test_changed_review_image_cannot_be_published(self):
        path = self.package/'previews/web-hero.png'
        path.write_bytes(b'changed after review')
        result = subprocess.run([sys.executable, ROOT/'tools/end/pipeline.py', 'package',
                                 'pipeline-fixture', '--package', self.package],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Acceptance evidence changed', result.stderr)


if __name__ == '__main__': unittest.main(verbosity=2)
