"""Verify complete export equality plus corrupt/unwritable optional cache recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--campaign', type=Path, required=True, help='completed three-pair profile-startup output')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--pen', type=Path, default=Path('target/release/pen'))
parser.add_argument('--project', type=Path, required=True)
args = parser.parse_args()
campaign, output = args.campaign.resolve(), args.out.resolve()
pen, project = args.pen.resolve(), args.project.resolve()
output.mkdir(parents=True, exist_ok=False)
sha = lambda data: hashlib.sha256(data).hexdigest()

def tree(root):
    return {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file()}

report = json.loads((campaign/'report.json').read_text())
assert len(report['samples']) == 24
assert sha(pen.read_bytes()) == report['host_pins'][str(pen)]
bundles = sorted(campaign.glob('*-*/bundle-*'))
assert len(bundles) == 12
expected = tree(bundles[0])
assert expected
for bundle in bundles:
    assert tree(bundle) == expected, f'Export file bytes differ: {bundle}'

records = []
def export(name, cache):
    destination = output / name
    env = dict(os.environ, XDG_CACHE_HOME=str(cache))
    command = [str(pen), 'export', '--project', str(project), '--out', str(destination)]
    started = time.monotonic()
    result = subprocess.run(command, env=env, capture_output=True, timeout=120)
    (output/(name+'.stdout')).write_bytes(result.stdout)
    (output/(name+'.stderr')).write_bytes(result.stderr)
    assert result.returncode == 0, f'{name}: export failed; see {output}'
    assert tree(destination) == expected, f'{name}: recovery changed exported bytes'
    records.append({'name':name, 'wall_seconds':time.monotonic()-started, 'returncode':result.returncode})

cache = (campaign/'0-candidate/export-cache').resolve()
assert cache.is_relative_to(campaign), 'Cache must belong to the isolated campaign'
artifacts = list((cache/'pen/wasm').glob('*.cwasm'))
assert len(artifacts) == 1, artifacts
artifact = artifacts[0].resolve()
assert artifact.is_relative_to(campaign), 'Artifact must belong to the isolated campaign'
original = artifact.read_bytes()
magic = b'PENCWASM\x01'
checksum_at = len(magic) + 64
header_bytes = checksum_at + 32
assert original.startswith(magic) and len(original)>header_bytes
corrupt = bytearray(original)
corrupt[len(corrupt)//2] ^= 1
artifact.write_bytes(corrupt)
export('corrupt-recovery', cache)
repaired = artifact.read_bytes()
assert repaired != bytes(corrupt)
assert repaired[:checksum_at] == original[:checksum_at]
assert hashlib.sha256(repaired[header_bytes:]).digest() == repaired[checksum_at:header_bytes]
export('warm-after-recovery', cache)

blocked = output/'blocked-cache'
shutil.copytree(cache, blocked, ignore=lambda directory,names: ['wasm'] if Path(directory).name=='pen' else [])
(blocked/'pen/wasm').write_bytes(b'not a directory')
export('unwritable-cache-fallback', blocked)
assert (blocked/'pen/wasm').read_bytes() == b'not a directory'
summary = {'pen_sha256':sha(pen.read_bytes()), 'probe_sha256':sha(Path(__file__).read_bytes()), 'exported_files':expected, 'equal_original_bundles':len(bundles), 'checks':records, 'repaired_payload_checksum':True}
(output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'equal_bundles':len(bundles),'files_per_bundle':len(expected),'checks':records}),flush=True)
