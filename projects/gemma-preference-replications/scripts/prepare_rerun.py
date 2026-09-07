"""Create a new experiment workspace with frozen fits/pairs but no trial outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('destination', type=Path, help='Must not already exist')
args = ap.parse_args()
dest = args.destination.resolve()
if dest.exists():
    ap.error(f'Destination exists: {dest}')
dest.mkdir(parents=True)
shutil.copytree(ROOT / 'scripts', dest / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
for name in ['requirements-analysis.txt', 'requirements-gpu.txt', '.gitignore', 'REPRODUCING.md',
             'PROVENANCE.md', 'LICENSE', 'LICENSE-DATA', 'THIRD_PARTY.md']:
    shutil.copy2(ROOT / name, dest / name)
inputs = ['runs/pairwise/pool.json', 'runs/pairwise/pairs.json', 'runs/pairwise/utilities.csv',
          'runs/extract/norms.json', 'runs/probe/directions.npy', 'runs/steer/pairs.json',
          'runs/null/dirs.npz', 'runs/null/natsd_scale.json']
manifest = []
for name in inputs:
    target = dest / name; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
    manifest.append({'path': name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
script_hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((dest / 'scripts').glob('*')) if p.is_file()}
(dest / 'STARTING_INPUTS.json').write_text(json.dumps({'inputs': manifest, 'code_sha256': script_hashes}, indent=2) + '\n')
print(f'Prepared {dest}. No model was loaded or experiment started.')
