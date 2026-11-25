"""Download two independently collected CC BY 4.0 water-leak datasets."""
from pathlib import Path
import concurrent.futures, hashlib, json, subprocess, zipfile
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data'

def get_json(url):
    r = requests.get(url, timeout=90); r.raise_for_status(); return r.json()

def fetch(job):
    project, name, url, md5 = job
    folder = RAW / project; folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    if not dest.exists():
        with requests.get(url, stream=True, timeout=(30, 180)) as r:
            r.raise_for_status()
            with dest.open('wb') as f:
                for chunk in r.iter_content(1024*1024): f.write(chunk)
    actual = hashlib.md5(dest.read_bytes()).hexdigest()
    assert actual == md5, (name, actual, md5)
    target = folder / dest.stem; target.mkdir(exist_ok=True)
    if dest.suffix == '.zip':
        with zipfile.ZipFile(dest) as z: z.extractall(target)
    else:
        subprocess.run(['tar', '-xf', str(dest), '-C', str(target)], check=True)
    print(project, name, dest.stat().st_size, 'checksum verified, extracted', flush=True)
    return dict(project=project, filename=name, url=url, md5=actual, bytes=dest.stat().st_size)

def main():
    ROOT.mkdir(exist_ok=True)
    z = get_json('https://zenodo.org/api/records/18631450')
    s = get_json('https://api.figshare.com/v2/articles/32229270/versions/2')
    (ROOT/'source_metadata.json').write_text(json.dumps({'dongguan':z, 'sheffield':s}, indent=2), encoding='utf-8')
    jobs = [('dongguan', f['key'], f['links']['self'], f['checksum'].split(':')[1]) for f in z['files']]
    jobs += [('sheffield', f['name'], f['download_url'], f['computed_md5']) for f in s['files']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(fetch, jobs))
    (ROOT/'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for project in ['dongguan','sheffield']:
        files = [p for p in (RAW/project).rglob('*') if p.is_file() and p.suffix not in ['.zip','.rar']]
        print(project, len(files), 'files')
        print('\n'.join(str(p.relative_to(RAW)) for p in files[:25]))

if __name__ == '__main__': main()
