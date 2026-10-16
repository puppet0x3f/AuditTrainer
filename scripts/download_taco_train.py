"""Download only the pinned TACO train Parquet files and verify their hashes."""

from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import HfApi
from tqdm import tqdm


REVISION = 'd593ed0a2becbbc952230bb89be09189bf1056dc'
ROOT = Path('data/taco_source')


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _download(item):
    name, size, sha256 = item
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == size and _hash(path) == sha256:
        return
    partial = path.with_suffix(path.suffix + '.part')
    if partial.exists() and partial.stat().st_size > size:
        partial.unlink()
    url = f'https://huggingface.co/datasets/BAAI/TACO/resolve/{REVISION}/{name}?download=true'
    for attempt in range(3):
        command = ['curl', '-L', '--fail', '--silent', '--show-error', '--retry', '3',
                   '--connect-timeout', '20', '--max-time', '1800', '-C', '-',
                   '-o', str(partial), url]
        completed = subprocess.run(command, capture_output=True, text=True)
        if (completed.returncode == 0 and partial.stat().st_size == size
                and _hash(partial) == sha256):
            partial.replace(path)
            return
        if partial.exists() and partial.stat().st_size == size:
            partial.unlink()
    raise RuntimeError(f'Could not verify TACO train file: {name}')


def main():
    info = HfApi().dataset_info('BAAI/TACO', revision=REVISION, files_metadata=True)
    items = sorted((item.rfilename, item.size, item.lfs.sha256)
                   for item in info.siblings
                   if item.rfilename.startswith('ALL/train-') and
                   item.rfilename.endswith('.parquet'))
    if len(items) != 9 or info.sha != REVISION:
        raise ValueError('Pinned TACO train revision or shard list changed')
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool, tqdm(
            total=sum(size for _, size, _ in items), unit='B', unit_scale=True,
            desc='TACO train 下载校验', dynamic_ncols=True) as bar:
        futures = [pool.submit(_download, item) for item in items]
        last = 0
        while True:
            current = sum(min(size, (ROOT / name).stat().st_size
                              if (ROOT / name).exists() else
                              (ROOT / name).with_suffix('.parquet.part').stat().st_size
                              if (ROOT / name).with_suffix('.parquet.part').exists() else 0)
                          for name, size, _ in items)
            bar.update(max(0, current - last))
            last = current
            done, pending = wait(futures, timeout=1)
            if not pending:
                for future in done:
                    future.result()
                break
        bar.update(bar.total - bar.n)
    (ROOT / 'manifest.json').write_text(json.dumps({
        'dataset': 'BAAI/TACO', 'revision': REVISION,
        'files': [{'path': name, 'bytes': size, 'sha256': sha256}
                  for name, size, sha256 in items]}, indent=2) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'train_shards': len(items),
                      'bytes': sum(size for _, size, _ in items)},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
