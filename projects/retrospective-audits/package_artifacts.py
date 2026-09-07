"""Package the recorded experiments for offline reanalysis; never reads Codex homes.

Run from this project directory: python3 package_artifacts.py
The archive contains byte-for-byte copies, selected by path below. It excludes
redundant workspace copies, credentials, and unrelated diagnostic runs.
"""
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SECRET = re.compile(rb'"(?:access_token|refresh_token|id_token|OPENAI_API_KEY)"\s*:\s*"[^"\s]+"')


def selected_files(run, followups):
    paths = [run / "result.json", run / "eligibility.json"]
    for folder in ("parent", "prompts", "ground_truth", "environment", "task", "fork_point", "verifier"):
        paths.extend(p for p in (run / folder).rglob("*") if p.is_file())
    phases = [run] + ([run / "belief", run / "debias"] if followups else [])
    for phase in phases:
        if phase != run:
            paths.extend(phase / name for name in ("result.json", "source_hashes.json", "nonce.txt"))
            for folder in ("prompts", "environment"):
                paths.extend(p for p in (phase / folder).rglob("*") if p.is_file())
        # Manifests retain the before/after comparison; omit duplicate workspaces.
        for branch in (phase / "branches").iterdir():
            if branch.is_dir():
                paths.extend(p for p in branch.iterdir() if p.is_file())
    return sorted(set(paths))


def main():
    archive = ROOT / "data" / "replication-artifacts.tar.gz"
    manifest_path = ROOT / "data" / "artifact-manifest.json"
    if archive.exists() or manifest_path.exists():
        raise SystemExit("Refusing to overwrite an existing artifact package.")
    paths = []
    for schedule, followups in (("schedule-main.json", True), ("schedule.json", False)):
        entries = json.loads((ROOT / "tasks" / schedule).read_text())["sequence"]
        for entry in entries:
            paths.extend(selected_files(ROOT / entry["run_dir"], followups))
    files = {}
    for path in sorted(set(paths)):
        if path.is_symlink() or path.name in {"auth.json", ".env", "config.toml"}:
            raise SystemExit(f"Unexpected private file or symlink: {path}")
        content = path.read_bytes()
        if SECRET.search(content):
            raise SystemExit(f"Credential field found: {path}")
        files[path.relative_to(ROOT).as_posix()] = content
    manifest = {"description": "Byte-preserving selection from 16 main and 22 pilot runs; generated after collection.",
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    archive.parent.mkdir(exist_ok=True)
    # Fixed metadata makes rebuilding the same raw selection byte-identical.
    with archive.open("xb") as handle, gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name, content in files.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(content), 0o644
                tar.addfile(info, io.BytesIO(content))
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Packaged {len(files)} files; {archive.stat().st_size:,} compressed bytes.")


if __name__ == "__main__":
    main()
