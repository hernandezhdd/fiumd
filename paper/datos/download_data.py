"""Download the benchmark inputs for cln, trp, bbl and lambda.

    cd paper/datos && python download_data.py            # all four proteins
    python download_data.py --molecules cln trp           # some of them
    python download_data.py --outdir /some/other/datos    # elsewhere

Source: Majewski et al., Nat. Commun. 14, 5739 (2023).
  - github.com/torchmd/torchmd-protein-thermodynamics (prior, starting clusters)
  - pub.htmd.org/protein_thermodynamics_data (embeddings, coordinates, Models.zip)

The .psf files are not downloaded: the ones in git have the bead masses set to
12.0 (upstream has 0.0, which gives infinite initial velocities).

Per protein this writes:
  <m>_prior_force_field.yaml           same file for all four proteins
  <m>_kcenters_32clusters_coords.xtc   32 starting structures
  <long>_ca_embeddings.npy             bead types
  <long>_5120_ca_coords.npy            pool of <=5120 CA positions (see POOLS)
  model.ckpt                           from Models.zip

The pool is cut from <long>_ca_coords.npy (0.2-13.8 GB) by reading only the
frames it needs with HTTP Range requests, so the whole file is never downloaded.
Each protein keeps the selection rule used for the paper, so the files come out
byte-identical to the ones listed in README.md (MD5).

Afterwards, from paper/:
    python benchmark/check_benchmark_data.py --fix --deep
writes model_modified.ckpt and checks that every file opens.
"""
import argparse
import hashlib
import http.client
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import numpy.lib.format as nf

HERE = Path(__file__).resolve().parent


GH = ("https://raw.githubusercontent.com/torchmd/torchmd-protein-thermodynamics/"
      "39beeab6c8399763c8304922f8ff2d11ee0962ec/Tutorials")
HTMD = "http://pub.htmd.org/protein_thermodynamics_data"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

POOL_MAX = 5120

# key -> (long name, number of beads, pool rule)
#   'arange':   frames np.arange(0, n, n // k)[:k]
#   'random42': frames sorted(np.random.seed(42); choice(n, k, replace=False))
#   'linspace': frames np.linspace(0, n - 1, k)
# with k = POOL_MAX // nbeads frames
MOLECULES = {
    "cln":    ("chignolin", 10, "arange"),
    "trp":    ("trpcage",   20, "arange"),
    "bbl":    ("bbl",       47, "random42"),
    "lambda": ("lambda",    80, "linspace"),
}


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if data.lstrip()[:20].lower().startswith(b"<!doctype html"):
        raise RuntimeError(f"{url} returned an HTML page, not the file")
    return data


def write(path, data):
    path.write_bytes(data)
    print(f"  -> {path.name} ({len(data) / 1024:.1f} KB)")


class RangeReader:
    """Byte ranges of one remote file over a keep-alive connection.

    pub.htmd.org closes the connection every ~20 requests, so a dropped
    connection is reopened and the request retried. A status other than 206
    is fatal: it would mean the server is sending the whole file.
    """

    def __init__(self, url):
        parts = urlsplit(url)
        self.cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        self.host, self.path, self.url = parts.netloc, parts.path, url
        self.conn = self.cls(self.host, timeout=60)

    def read(self, start, end, tries=5):
        for attempt in range(tries):
            try:
                self.conn.request("GET", self.path, headers={**HEADERS, "Range": f"bytes={start}-{end}"})
                r = self.conn.getresponse()
                data = r.read()
                if r.status != 206:
                    raise RuntimeError(f"{self.url} ignored the Range header (status {r.status})")
                return data
            except (http.client.HTTPException, OSError) as e:
                self.conn.close()
                self.conn = self.cls(self.host, timeout=60)
                if attempt == tries - 1:
                    raise RuntimeError(f"bytes {start}-{end} of {self.url}: {e}") from e
                time.sleep(0.5 * (attempt + 1))

    def close(self):
        self.conn.close()


def frame_indices(rule, n_frames, k):
    if rule == "arange":
        return np.arange(0, n_frames, n_frames // k)[:k]
    if rule == "random42":
        np.random.seed(42)
        return np.sort(np.random.choice(n_frames, size=k, replace=False))
    if rule == "linspace":
        return np.linspace(0, n_frames - 1, k).astype(np.int64)
    raise ValueError(rule)


def download_pool(long_name, nbeads, rule, outdir):
    url = f"{HTMD}/training_data/{long_name}_ca_coords.npy"
    reader = RangeReader(url)
    try:
        header = io.BytesIO(reader.read(0, 255))
        version = nf.read_magic(header)
        read_header = {(1, 0): nf.read_array_header_1_0, (2, 0): nf.read_array_header_2_0}[version]
        shape, fortran, dtype = read_header(header)
        offset = header.tell()
        if fortran or shape[1] != nbeads:
            raise RuntimeError(f"{url}: unexpected layout {shape}, fortran={fortran}")

        k = POOL_MAX // nbeads
        idx = frame_indices(rule, shape[0], k)
        frame_bytes = nbeads * 3 * dtype.itemsize
        print(f"  {long_name}_ca_coords.npy: {shape[0]} frames, reading {len(idx)} ({rule})")
        chunks = []
        for j, i in enumerate(idx, 1):
            start = offset + int(i) * frame_bytes
            chunks.append(reader.read(start, start + frame_bytes - 1))
            sys.stdout.write(f"\r  {j}/{len(idx)} frames")
            sys.stdout.flush()
        print()
    finally:
        reader.close()

    pool = np.frombuffer(b"".join(chunks), dtype=dtype).reshape(-1, nbeads, 3)
    bond = np.linalg.norm(pool[:, 1:] - pool[:, :-1], axis=-1).mean()
    if not 2.5 <= bond <= 5.5:
        raise RuntimeError(f"mean CA-CA distance {bond:.2f} A: the frames are not physical")
    name = f"{long_name}_5120_ca_coords.npy"
    np.save(outdir / name, np.ascontiguousarray(pool.reshape(-1, 3)))
    print(f"  -> {name} {pool.shape[0] * nbeads} positions, mean CA-CA {bond:.2f} A")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--molecules", nargs="+", choices=list(MOLECULES), default=list(MOLECULES))
    p.add_argument("--outdir", default=str(HERE), help="folder with one subfolder per protein")
    args = p.parse_args()
    root = Path(args.outdir)

    print("=== prior force field and Models.zip ===")
    prior = get(f"{GH}/prior_force_field.yaml")
    models = zipfile.ZipFile(io.BytesIO(get(f"{HTMD}/Models.zip")))

    failed = []
    for key in args.molecules:
        long_name, nbeads, rule = MOLECULES[key]
        out = root / key
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {key} ({long_name}, {nbeads} beads) ===")
        try:
            write(out / f"{key}_prior_force_field.yaml", prior)
            write(out / f"{key}_kcenters_32clusters_coords.xtc",
                  get(f"{GH}/starting_xtc/{key}_kcenters_32clusters_coords.xtc"))
            write(out / f"{long_name}_ca_embeddings.npy",
                  get(f"{HTMD}/training_data/{long_name}_ca_embeddings.npy"))
            write(out / "model.ckpt", models.read(f"Models/{key}/model.ckpt"))
            download_pool(long_name, nbeads, rule, out)
        except Exception as e:  # report and go on with the next protein
            print(f"  !! {type(e).__name__}: {e}")
            failed.append(key)

    print("\nMD5:")
    for f in sorted(root.glob("*/*")):
        if f.suffix in {".yaml", ".xtc", ".npy", ".ckpt"}:
            print(f"{hashlib.md5(f.read_bytes()).hexdigest()}  {f.relative_to(root)}")
    if failed:
        print("\nFailed:", ", ".join(failed))
    print("\nNext, from paper/:  python benchmark/check_benchmark_data.py --fix --deep")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())