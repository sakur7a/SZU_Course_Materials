"""M0 command line: isolated build, independent checks, preview, portable package.

Builds remain isolated. The explicit integrate command installs only the reviewed
Lide runtime projection; it does not regenerate the full campus.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
from asset_checks import check_package, inside, sha256

ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT/'main/end/modeling'
BUILD = ROOT/'build/modeling-pipeline'
VIEWER = MODELING/'viewer'
SCHEMAS = MODELING/'schemas'
PROBE = BUILD/'cmake/bin/campus-asset-probe/campus-asset-probe.exe'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def run(command, *, cwd=ROOT, log=None, env=None, timeout=240):
    result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding='utf-8', errors='replace', timeout=timeout)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {command[0]}\n{result.stdout[-6000:]}')
    return result.stdout


def blender():
    candidates = [os.environ.get('BLENDER_EXE'), shutil.which('blender'),
                  ROOT/'build/campus-opensource-review/blender-pilot/blender-4.5.3-windows-x64/blender.exe']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise RuntimeError('Blender not found. Set BLENDER_EXE to blender.exe.')


def asset_dir(asset):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', asset):
        raise ValueError('asset-id must contain lowercase letters, digits and hyphens')
    return MODELING/'assets'/asset


def resolve_package(args, *, validated=False):
    asset_dir(args.asset)
    if getattr(args, 'package', None): return Path(args.package).resolve()
    pointer = BUILD/args.asset/('validated.json' if validated else 'latest.json')
    if not pointer.is_file(): raise RuntimeError(f'Run build/validate first: {pointer}')
    return inside(BUILD/args.asset, read(pointer)['run'])


def doctor(_):
    try:
        from importlib.metadata import version
        schema_version = version('jsonschema')
    except ImportError:
        schema_version = None
    info = {'python': sys.version.split()[0], 'jsonschema': schema_version,
            'blender': run([blender(), '--version']).splitlines()[0],
            'node': run(['node', '--version']).strip(),
            'viewer_dependencies': (VIEWER/'node_modules/three').is_dir(),
            'cpp_probe': PROBE.is_file(),
            'campus_integration': 'validated lide-gate runtime projection; other assets remain separate'}
    write(BUILD/'doctor.json', info)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not schema_version: raise RuntimeError('Install tools/end/requirements-pipeline.txt in the dedicated venv')


def build(args):
    from jsonschema import Draft202012Validator
    source_dir = asset_dir(args.asset)
    config = read(source_dir/'asset.json')
    Draft202012Validator(read(SCHEMAS/'asset.schema.json')).validate(config)
    if config['asset_id'] != args.asset: raise ValueError('Asset identity mismatch')
    if config['source_mode'] != 'procedural':
        raise NotImplementedError('M0 supports procedural fixtures; blend-authoritative build comes with M1')
    entry = inside(source_dir, config['source'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = BUILD/args.asset/stamp
    output.mkdir(parents=True)
    inputs = {p.relative_to(ROOT).as_posix(): sha256(p) for folder in [source_dir, ROOT/'tools/end/blender']
              for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    executable = blender()
    version = run([executable, '--version']).splitlines()[0]
    run([executable, '--background', '--factory-startup', '--python-exit-code', '1',
         '--python', entry, '--', '--output', output], log=output/'logs/build.log')
    manifest = read(output/'manifest.json')
    if (source_dir/'references').is_dir():
        shutil.copytree(source_dir/'references', output/'references')
    context_file = source_dir/'context/context.json'
    if context_file.is_file():
        context = read(context_file)
        run([executable, '--background', '--factory-startup', '--python-exit-code', '1',
             '--python', ROOT/'tools/end/blender/export_context.py', '--', '--context', context_file,
             '--output', output/'context.glb', '--report', output/'checks/context.json'],
            log=output/'logs/context.log')
        manifest['preview_context'] = {'model': 'context.glb', 'role': 'read_only_surroundings',
                                       'source_sha256': sha256(context_file)}
        manifest['files']['context.glb'] = sha256(output/'context.glb')
        manifest['placement'] = {'translation': context['origin_yup'], 'rotation': [0, 0, 0, 1], 'scale': [1, 1, 1]}
        placement = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, *context['origin_yup'], 1]
        (output/'placement.tsv').write_text('P\t'+'\t'.join(format(x, '.12g') for x in placement)+'\n', encoding='utf-8')
        manifest['files']['placement.tsv'] = sha256(output/'placement.tsv')
    manifest['provenance'] = {'built_at_utc': stamp, 'blender': version,
                              'host_python': sys.version.split()[0], 'inputs': inputs}
    write(output/'manifest.json', manifest)
    report = check_package(output, SCHEMAS)
    write(output/'checks/geometry.json', report)
    write(BUILD/args.asset/'latest.json', {'run': stamp})
    print(f'BUILT {output}\nGeometry checks passed; final validation is still required.')


def probe_build(_):
    presets = read(ROOT/'CMakePresets.json')['configurePresets']
    preset = next(p for p in presets if p['name'] == 'mingw-vcpkg')
    command = ['cmake', '-S', ROOT, '-B', BUILD/'cmake', '-G', preset['generator'],
               '-DCAMPUS_PIPELINE_ONLY=ON']
    for name, value in preset['cacheVariables'].items():
        if name != 'CMAKE_BUILD_TYPE': command.append(f'-D{name}={value}')
    command.append('-DCMAKE_BUILD_TYPE=Release')
    run(command, log=BUILD/'cmake-configure.log')
    run(['cmake', '--build', BUILD/'cmake', '--target', 'campus-asset-probe', '-j', '4'],
        log=BUILD/'cmake-build.log')
    print(PROBE)


def validate(args):
    from PIL import Image, ImageChops
    output = resolve_package(args)
    report = check_package(output, SCHEMAS)
    previous_native = read(output/'checks/native.json') if (output/'checks/native.json').exists() else None
    run(['node', VIEWER/'scripts/validate-glb.mjs', output/'model.glb', output/'checks/gltf.json'],
        log=output/'logs/gltf.log')
    if read(output/'manifest.json').get('preview_context'):
        run(['node', VIEWER/'scripts/validate-glb.mjs', output/'context.glb', output/'checks/context-gltf.json'],
            log=output/'logs/context-gltf.log')
    run([blender(), '--background', output/'model.blend', '--python-exit-code', '1',
         '--python', ROOT/'tools/end/blender/reopen_verify.py', '--', '--package', output,
         '--report', output/'checks/reopen.json'], log=output/'logs/reopen.log')
    if not PROBE.is_file(): raise RuntimeError('Run probe-build before validate')
    run([PROBE, '--asset', output, '--self-test', '--report', output/'checks/cpp.json'],
        log=output/'logs/cpp.log')
    # Rendering evidence is produced by the real OpenGL path, separately from self-test.
    (output/'previews').mkdir(exist_ok=True)
    captures = {}
    max_angle = read(output/'manifest.json')['interaction']['range_deg'][1]
    capture_views = [('hero', 'hero', 0), ('front', 'front', 0), ('top', 'top', 0), ('open', 'hero', max_angle)]
    if 'walk' in read(output/'manifest.json')['views']: capture_views.append(('walk', 'walk', 0))
    for name, view, angle in capture_views:
        ppm = output/f'previews/native-{name}.ppm'
        png = ppm.with_suffix('.png')
        run([PROBE, '--asset', output, '--view', view, '--angle', str(angle), '--capture', ppm],
            log=output/f'logs/native-{name}.log')
        with Image.open(ppm) as frame:
            if frame.size != (1280, 800): raise ValueError('Unexpected native framebuffer dimensions')
            frame.save(png)
        captures[png.name] = sha256(png)
    with Image.open(output/'previews/native-hero.png') as closed, Image.open(output/'previews/native-open.png') as opened:
        if ImageChops.difference(closed, opened).getbbox() is None:
            raise ValueError('Native hinge action did not change rendered pixels')
    native = {'status': 'passed', 'manifest_sha256': report['manifest_sha256'],
              'cpp_executable_sha256': sha256(PROBE), 'captures': captures}
    write(output/'checks/native.json', native)
    report['automated_core'] = 'passed'
    report['cpp_executable_sha256'] = sha256(PROBE)
    report['visual_review'] = 'pending'
    browser_report = output/'checks/browser.json'
    if args.visual_reviewed:
        if not previous_native or previous_native != native:
            raise RuntimeError('Native capture evidence changed or was missing; inspect the new captures before accepting')
        if not browser_report.is_file(): raise RuntimeError('Browser QA report is required before visual acceptance')
        browser = read(browser_report)
        if browser.get('status') != 'passed' or browser.get('manifest_sha256') != sha256(output/'manifest.json'):
            raise RuntimeError('Browser QA must pass against this exact manifest')
        if not list((output/'previews').glob('*.png')):
            raise RuntimeError('Review evidence screenshots are missing')
        evidence = [output/'checks/native.json', output/'checks/browser.json',
                    *[output/'previews'/name for name in native['captures']],
                    *[output/'previews'/name for name in browser['screenshots']]]
        for optional in ['native-ui.json', 'site.json', 'context.json', 'context-gltf.json']:
            candidate = output/'checks'/optional
            if candidate.is_file():
                if optional == 'native-ui.json':
                    ui = read(candidate)
                    if ui.get('cpp_executable_sha256') != sha256(PROBE) or ui.get('manifest_sha256') != report['manifest_sha256']:
                        report['native_ui_review'] = 'historical_report_not_bound_to_current_executable'
                        continue
                evidence.append(candidate)
        report['evidence'] = {p.relative_to(output).as_posix(): sha256(p) for p in evidence}
        report['visual_review'] = 'reviewed'
        report['status'] = 'accepted'
        write(BUILD/args.asset/'validated.json', {'run': output.name})
    write(output/'checks/validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def web_build(output, destination, public):
    # Fresh staging directory: no source assets are overwritten or deleted.
    target = public/'assets'/read(output/'manifest.json')['asset_id']
    if target.exists(): raise ValueError(f'Staging directory already exists: {target}')
    shutil.copytree(output, target, ignore=shutil.ignore_patterns('logs', '*.ppm'))
    env = os.environ.copy()
    env['PIPELINE_PUBLIC_DIR'] = str(public)
    env['PIPELINE_WEB_OUT'] = str(destination)
    npm = shutil.which('npm.cmd' if os.name == 'nt' else 'npm')
    if not npm: raise RuntimeError('npm is unavailable')
    run([npm, 'run', 'build'], cwd=VIEWER, env=env, timeout=240,
        log=output/'logs/web-build.log')


def preview(args):
    output = resolve_package(args)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    staging = BUILD/'web-previews'/stamp
    web_build(output, staging/'web', staging/'public')
    print(f'http://127.0.0.1:{args.port}/?asset=/assets/{args.asset}/manifest.json', flush=True)
    if args.prepare_only:
        print(staging/'web')
        return
    subprocess.run([sys.executable, '-m', 'http.server', str(args.port), '--bind', '127.0.0.1',
                    '--directory', str(staging/'web')], check=True)


def package(args):
    output = resolve_package(args, validated=True)
    check_package(output, SCHEMAS)
    asset_manifest = read(output/'manifest.json')
    validation = read(output/'checks/validation.json')
    if validation.get('status') != 'accepted' or validation['manifest_sha256'] != sha256(output/'manifest.json'):
        raise RuntimeError('Only the exact accepted output can be packaged')
    if validation['cpp_executable_sha256'] != sha256(PROBE):
        raise RuntimeError('C++ executable changed since validation')
    for relative, expected in validation['evidence'].items():
        if sha256(inside(output, relative)) != expected:
            raise RuntimeError(f'Acceptance evidence changed: {relative}')
    release_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    release = BUILD/'releases'/f'{args.asset}-{release_stamp}'
    release.mkdir(parents=True, exist_ok=False)
    shutil.copytree(output, release/'asset', ignore=shutil.ignore_patterns('logs', '*.ppm'))
    # Preserve the builder's relative import layout so the distributed source rebuilds independently.
    shutil.copytree(asset_dir(args.asset), release/'source/main/end/modeling/assets'/args.asset,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'tools/end/blender', release/'source/tools/end/blender',
                    ignore=shutil.ignore_patterns('__pycache__'))
    if (asset_dir(args.asset)/'context/context.json').is_file():
        context = read(asset_dir(args.asset)/'context/context.json')
        (release/'CREDITS.txt').write_text(context['provenance']['attribution']+'\n\n'
            'OpenStreetMap: https://www.openstreetmap.org/copyright\n'
            'Terrain: https://registry.opendata.aws/terrain-tiles/\n'
            'Terrain attribution: https://github.com/tilezen/joerd/blob/master/docs/attribution.md\n'
            'Grass004 (CC0): https://ambientcg.com/view?id=Grass004\n'
            'Reference photographs are not included.\n', encoding='utf-8')
    native = release/'native'; native.mkdir()
    shutil.copy2(PROBE, native/PROBE.name)
    (release/'open-native.cmd').write_text('@echo off\ncd /d "%~dp0"\n"native\\campus-asset-probe.exe" --asset "%~dp0asset"\n', encoding='ascii')
    (release/'open-web.cmd').write_text('@echo off\ncd /d "%~dp0"\npython start-preview.py\n', encoding='ascii')
    (release/'start-preview.py').write_text(
        'from functools import partial\nfrom http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler\n'
        'from pathlib import Path\nimport webbrowser\n'
        'root = Path(__file__).resolve().parent\n'
        'server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root / "web")))\n'
        f'url = f"http://127.0.0.1:{{server.server_port}}/?asset=/assets/{args.asset}/manifest.json"\n'
        'print(url, flush=True)\nwebbrowser.open(url)\n'
        'try:\n    server.serve_forever()\nexcept KeyboardInterrupt:\n    pass\nfinally:\n    server.server_close()\n', encoding='utf-8')
    web_build(output, release/'web', BUILD/'package-staging'/release_stamp)
    for relative, expected in read(output/'checks/browser.json')['resources'].items():
        if sha256(inside(release/'web', relative)) != expected:
            raise RuntimeError(f'Viewer changed since browser validation: {relative}')
    (release/'README.txt').write_text(
        (('Synthetic modeling-pipeline fixture; not a real campus reconstruction.\n') if asset_manifest.get('synthetic')
         else ('Photo-based approximate landmark; dimensions and terrain are not surveyed.\n'))+
        'Native: double-click open-native.cmd. A/D or left/right: hinge; 1/2/3: views; Esc: exit.\n'
        'Web: double-click open-web.cmd (Python 3 required); it opens the asset in your browser on a free local port.\n'
        'The native probe shows the asset body; optional read-only surroundings are shown separately in the web viewer.\n'
        'Editable scene: asset/model.blend. Runtime asset: asset/model.glb.\n'
        f'Rebuild with installed Blender: blender -b --factory-startup --python-exit-code 1 --python source/main/end/modeling/assets/{args.asset}/source/build.py -- --output rebuilt\n', encoding='utf-8')
    run([native/PROBE.name, '--asset', release/'asset', '--self-test'], cwd=release)
    hashes = {p.relative_to(release).as_posix(): sha256(p) for p in release.rglob('*') if p.is_file()}
    write(release/'release.json', {'asset_id': args.asset, 'manifest_sha256': validation['manifest_sha256'], 'files': hashes})
    archive = release.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in release.rglob('*'):
            if p.is_file(): z.write(p, p.relative_to(release).as_posix())
    write(BUILD/args.asset/'latest-release.json', {'directory': str(release.relative_to(BUILD)),
                                                'archive': str(archive.relative_to(BUILD))})
    print(f'RELEASE {release}\nZIP {archive}')


def integrate(args):
    """Install only an accepted landmark's runtime projection; preserve campus base layers."""
    if args.scene != 'yuehai' or args.asset != 'lide-gate':
        raise NotImplementedError('Only the reviewed Lide replacement is implemented at this stage')
    output = resolve_package(args, validated=True)
    check_package(output, SCHEMAS)
    manifest = read(output/'manifest.json')
    acceptance = read(output/'checks/validation.json')
    if acceptance.get('status') != 'accepted' or acceptance['manifest_sha256'] != sha256(output/'manifest.json'):
        raise ValueError('Integration requires the exact visually accepted asset')
    for name, digest in acceptance['evidence'].items():
        if sha256(inside(output, name)) != digest: raise ValueError('Acceptance evidence changed')
    site = read(output/'checks/site.json')
    if site.get('status') not in ('passed', 'passed_with_limits') or site.get('issues'):
        raise ValueError('Site geometry/placement checks must pass first')
    if site.get('manifest_sha256') != sha256(output/'manifest.json'):
        raise ValueError('Site checks belong to a different asset version')
    context = read(asset_dir(args.asset)/'context/context.json')
    if site.get('context_sha256') != sha256(asset_dir(args.asset)/'context/context.json'):
        raise ValueError('Site checks belong to a different context snapshot')
    for layer in context['layers']:
        for source in (layer.get('source_mesh'), layer.get('source_texture')):
            if source and sha256(inside(ROOT, source['path'])) != source['sha256']:
                raise ValueError('Campus context changed; re-extract and revalidate before integrating')
    base = ROOT/'main/end/code/assets'
    managed = base/'modeling'; managed.mkdir(exist_ok=True)
    target = inside(managed, args.asset)
    if target.is_symlink(): raise ValueError('Refusing a symlink installation target')
    if target.exists() and not (target/'installation.json').is_file():
        raise ValueError('The destination is not a managed pipeline installation')
    untouched = {p: sha256(p) for p in base.rglob('*') if p.is_file() and target not in p.parents}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    staging = inside(managed, f'.{args.asset}-{stamp}')
    staging.mkdir()
    runtime = {name: digest for name, digest in manifest['files'].items()
               if name.startswith('legacy/') or name == 'placement.tsv'}
    if 'placement.tsv' not in runtime: raise ValueError('Missing campus placement')
    for name, digest in runtime.items():
        destination = inside(staging, name); destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(output, name), destination)
        if sha256(destination) != digest: raise ValueError('Runtime copy hash mismatch')
    installation = {'asset_id': args.asset, 'asset_version': manifest['asset_version'],
                    'source_manifest_sha256': acceptance['manifest_sha256'], 'files': runtime,
                    'placement': manifest['placement'], 'replaces': {'index': 'campus-gates.tsv', 'id': 'lide'},
                    'context_installed': False, 'original_terrain_or_roads_modified': False}
    write(staging/'installation.json', installation)
    backup = None
    if target.exists():
        backup = inside(BUILD, f'integration-backups/{args.asset}/{stamp}')
        backup.parent.mkdir(parents=True, exist_ok=True)
        # Both resolved move targets are verified inside the named source/build roots.
        target.rename(backup)
    try:
        staging.rename(target)
    except OSError:
        if backup: backup.rename(target)
        raise
    changed = [str(p.relative_to(ROOT)) for p, digest in untouched.items() if not p.exists() or sha256(p) != digest]
    if changed: raise RuntimeError(f'Unrelated campus resources changed: {changed}')
    scene_path = MODELING/'scenes/yuehai.scene.json'
    scene = read(scene_path) if scene_path.exists() else {'schema_version': 1, 'units': 'm', 'coordinate_system': 'RH_Y_UP', 'assets': []}
    scene['assets'] = [a for a in scene['assets'] if a['asset_id'] != args.asset]
    scene['assets'].append({'asset_id': args.asset, 'source_manifest_sha256': acceptance['manifest_sha256'],
                            'placement': manifest['placement'], 'replaces_legacy_gate': 'lide'})
    write(scene_path, scene)
    write(output/'checks/integration.json', {'status': 'installed_pending_runtime_verification',
           'runtime_directory': target.relative_to(ROOT).as_posix(), 'copied_files': len(runtime),
           'unchanged_original_files': len(untouched), 'changed_original_files': changed})
    print(f'INSTALLED {target}\nOriginal files unchanged: {len(untouched)}; rebuild and verify final-campus next.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor').set_defaults(func=doctor)
    commands.add_parser('probe-build').set_defaults(func=probe_build)
    integration = commands.add_parser('integrate')
    integration.add_argument('asset')
    integration.add_argument('--scene', required=True)
    integration.set_defaults(func=integrate)
    for name, fn in [('build', build), ('validate', validate), ('preview', preview), ('package', package)]:
        p = commands.add_parser(name)
        p.add_argument('asset')
        if name != 'build': p.add_argument('--package', help='explicit package directory to inspect')
        if name == 'validate': p.add_argument('--visual-reviewed', action='store_true', help='record completed human/agent image review after browser QA')
        if name == 'preview':
            p.add_argument('--port', type=int, default=4173)
            p.add_argument('--prepare-only', action='store_true')
        p.set_defaults(func=fn)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
