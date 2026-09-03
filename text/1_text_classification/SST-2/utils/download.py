"""Resumable multi-connection downloader (shared by the SST-2 and GloVe fetch).

Ported from the segmentation projects' dataset/voc.py, where it existed for
one reason: the hosts we depend on throttle PER CONNECTION, so from a far-away
network a single stream crawls and a 862 MB GloVe archive takes hours. Opening
N parallel HTTP byte-range requests multiplies the throughput by roughly N,
and because each segment is written with append mode, a dropped connection (or
a Ctrl-C) costs only the bytes still missing -- rerunning resumes.

Kept in utils/ rather than dataset/ because TWO different downloads use it
here (the corpus and the word vectors), unlike the CNN projects where the one
copy lived next to the one dataset.
"""

import hashlib
import math
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

# Default number of parallel byte-range connections. The speedup comes from the
# throttle being per-connection; beyond ~16 the returns vanish and servers
# start refusing.
DEFAULT_CONNECTIONS = 16


def md5_of(path: str) -> str:
    """md5 hex digest of a file, streamed in 1 MB blocks (constant memory)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _probe_url(url: str):
    """Ask the server for the file size and whether byte ranges are supported.

    Sends a GET with "Range: bytes=0-0" (a 1-byte request):
      * status 206 + "Content-Range: bytes 0-0/<total>" -> ranges supported,
        total parsed from after the "/";
      * status 200 -> the server ignored the Range header: single stream only,
        total = Content-Length.

    Output: (supports_ranges: bool, total_bytes: int).
    """
    req = urllib.request.Request(
        url, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status == 206:
            total = int(resp.headers["Content-Range"].rsplit("/", 1)[-1])
            return True, total
        return False, int(resp.headers.get("Content-Length", 0))


def _fetch_range(url, part_path, start, end, progress, retries: int = 3):
    """Download bytes [start, end] (inclusive) of `url` into `part_path`.

    Resumable: if part_path already holds k bytes from an earlier attempt, the
    request asks for bytes start+k..end, so finished bytes are NEVER fetched
    twice -- a dropped connection costs nothing but the retry.

    Input:
        url, part_path: source and destination of this one segment.
        start, end: absolute byte offsets of the segment inside the file.
        progress: callable(n_new_bytes), called per chunk (thread-safe --
            tqdm.update holds an internal lock).
        retries: how many times to re-open a dropped connection.
    """
    expected = end - start + 1
    last_err = None
    for _ in range(retries):
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if have > expected:
            # Leftover from a run with a different split layout: start over.
            os.remove(part_path)
            have = 0
        if have == expected:
            return
        try:
            req = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start + have}-{end}",
                         "User-Agent": "Mozilla/5.0"})
            # "ab" append mode is what makes the resume work: each retry
            # continues writing exactly where the last attempt stopped.
            with urllib.request.urlopen(req, timeout=60) as resp, \
                    open(part_path, "ab") as f:
                for chunk in iter(lambda: resp.read(1 << 16), b""):
                    f.write(chunk)
                    progress(len(chunk))
            if os.path.getsize(part_path) == expected:
                return
        except Exception as e:  # timeout / reset / 503 -> retry from `have`
            last_err = e
    raise RuntimeError(
        f"segment {os.path.basename(part_path)} incomplete "
        f"after {retries} attempts") from last_err


def download_segmented(url, dest, md5=None, connections: int = DEFAULT_CONNECTIONS):
    """Multi-connection resumable download of `url` into `dest`.

    How it works:
      1. Probe the server: total size + byte-range support (no ranges -> fall
         back to ONE resumable stream, still better than a plain urlretrieve).
      2. Split [0, total) into `connections` equal segments and download them
         in parallel threads into dest.part0..partN (each segment resumes its
         own partial file, so an interrupted run loses nothing).
      3. Concatenate the parts into `dest`, delete the parts, verify the md5.

    Input:
        url: source URL.
        dest: final file path.
        md5: expected hex digest, or None to skip verification (used for the
            GLUE zip, which publishes no official checksum).
        connections: number of parallel range requests.
    """
    # Already downloaded? With an md5 we can prove it; without one, a
    # non-empty file is taken at face value (the caller checks the extracted
    # contents afterwards anyway).
    if os.path.exists(dest):
        if md5 is None:
            print(f"  found existing {os.path.basename(dest)}, skipping download")
            return
        print(f"  found existing {os.path.basename(dest)}, checking md5...")
        if md5_of(dest) == md5:
            print("  already complete (md5 OK), skipping download")
            return
        print("  incomplete/corrupt -> removing and re-downloading")
        os.remove(dest)

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    supports_ranges, total = _probe_url(url)
    if total <= 0:
        raise RuntimeError(f"server reported no file size for {url}")
    if not supports_ranges:
        connections = 1  # degrade gracefully: one (still resumable) stream

    # Split into `connections` contiguous segments covering [0, total).
    part_size = math.ceil(total / connections)
    ranges = [(i * part_size, min((i + 1) * part_size, total) - 1)
              for i in range(connections)]
    part_paths = [f"{dest}.part{i}" for i in range(len(ranges))]

    # Bytes already on disk from a previous interrupted run (only for the
    # progress bar's starting point -- they will not be downloaded again).
    already = sum(
        min(os.path.getsize(p), e - s + 1)
        for p, (s, e) in zip(part_paths, ranges) if os.path.exists(p))

    try:
        from tqdm import tqdm
        bar = tqdm(total=total, initial=already, unit="B", unit_scale=True,
                   desc=f"  {connections} connections")
        progress = bar.update  # thread-safe (tqdm locks internally)
    except ImportError:
        bar = None

        def progress(n):  # no tqdm -> silent (sizes printed at the end)
            pass

    try:
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(_fetch_range, url, part, s, e, progress)
                for part, (s, e) in zip(part_paths, ranges)
            ]
            for f in futures:
                f.result()  # re-raise the first segment failure
    finally:
        if bar is not None:
            bar.close()

    # Concatenate the segments in order, then drop them.
    with open(dest, "wb") as out:
        for part in part_paths:
            with open(part, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    out.write(chunk)
    for part in part_paths:
        os.remove(part)

    if md5 is not None and md5_of(dest) != md5:
        os.remove(dest)
        raise RuntimeError(f"md5 mismatch for {dest} -- download corrupt")
    print(f"  downloaded {os.path.getsize(dest) / 1e6:.1f} MB -> {dest}")


def download_with_mirrors(urls, dest, md5=None, connections=DEFAULT_CONNECTIONS):
    """Try each mirror in turn until one delivers the file.

    Mirrors matter here for the same reason they did for VOC: the canonical
    academic hosts (nlp.stanford.edu, dl.fbaipublicfiles.com) go down or block
    whole regions for days at a time, while the HuggingFace copies stay up.

    Input:
        urls: list of candidate URLs, tried in order (put the fastest first).
        dest / md5 / connections: as in download_segmented.
    """
    last_err = None
    for url in urls:
        try:
            print(f"  trying {url}")
            download_segmented(url, dest, md5=md5, connections=connections)
            return
        except Exception as e:
            print(f"  FAILED ({type(e).__name__}: {e})")
            last_err = e
    raise RuntimeError(f"all mirrors failed for {os.path.basename(dest)}") from last_err


def extract_zip(archive: str, dest_dir: str, members=None):
    """Extract `archive` into `dest_dir`, optionally only `members`.

    Input:
        archive: path to the .zip file.
        dest_dir: destination directory (created if missing).
        members: list of archive member names to extract; None = everything.
            GloVe's zip holds four files (50d/100d/200d/300d, 2.0 GB
            unpacked) and we only need one -- passing members avoids
            unpacking 1.7 GB we would never read.
    """
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest_dir, members=members)
    print(f"  extracted -> {dest_dir}")
