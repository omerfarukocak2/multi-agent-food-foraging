"""Build an integrity manifest and a portable ZIP without environments or caches."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path.cwd().resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise RuntimeError('Run inside the project root directory.')
    excluded = {'tmp', '__pycache__', '.pytest_cache', '.venv', '.git'}
    files = sorted(p for p in root.rglob('*') if p.is_file() and
                   not any(part in excluded for part in p.relative_to(root).parts)
                   and p.name != 'MANIFEST.json')
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    output = root.parent / f'{root.name}.zip'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files + [root / 'MANIFEST.json']:
            archive.write(path, Path(root.name) / path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    print(f'Packaged {len(files) + 1} files: {output} ({output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
