"""
Download large Python wheels with retry + resume support,
then install them. Run this script from its directory.
"""
import os, time, subprocess, sys
import requests

WHEELS = [
    # (url, filename)
    (
        "https://files.pythonhosted.org/packages/cd/98/2cda2e6a0e63da5e7c11cd84cbf93a47dd918c3e94a77960d7f43b0c41c8/rasterio-1.5.2-cp312-cp312-win_amd64.whl",
        "rasterio-1.5.2-cp312-cp312-win_amd64.whl"
    ),
    (
        "https://files.pythonhosted.org/packages/b6/b2/46d2cc25d7bf6f4eb35a0b82fce7d14d9c2e40e0d1a4e0e80ee5a5c40a46/pyproj-3.8.0-cp312-cp312-win_amd64.whl",
        "pyproj-3.8.0-cp312-cp312-win_amd64.whl"
    ),
    (
        "https://files.pythonhosted.org/packages/f7/ce/72f9d6d17e67e073b7a48ca5d9c1f9c46ae76cf6898caa8f64b78b3a0c0d/geopandas-1.0.1-py3-none-any.whl",
        "geopandas-1.0.1-py3-none-any.whl"
    ),
    (
        "https://files.pythonhosted.org/packages/pyogrio/pyogrio-0.13.0-cp311-abi3-win_amd64.whl",
        "pyogrio-0.13.0-cp311-abi3-win_amd64.whl"
    ),
    (
        "https://files.pythonhosted.org/packages/affine/affine-3.0.1-py3-none-any.whl",
        "affine-3.0.1-py3-none-any.whl"
    ),
]

CHUNK = 512 * 1024  # 512 KB chunks
MAX_RETRIES = 10

def download_with_resume(url, dest):
    if os.path.exists(dest):
        existing = os.path.getsize(dest)
        print(f"  Found partial: {existing/1e6:.1f} MB")
    else:
        existing = 0

    headers = {}
    if existing > 0:
        headers['Range'] = f'bytes={existing}-'

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    print("  Already fully downloaded.")
                    return True
                r.raise_for_status()
                total = int(r.headers.get('Content-Length', 0)) + existing
                mode = 'ab' if existing else 'wb'
                with open(dest, mode) as f:
                    downloaded = existing
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pct = downloaded / total * 100 if total else 0
                            print(f"\r  {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)", end='', flush=True)
                            existing = downloaded
                            headers['Range'] = f'bytes={existing}-'
            print(f"\n  Download complete: {dest}")
            return True
        except Exception as e:
            print(f"\n  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(5)

    return False

os.makedirs("wheels", exist_ok=True)

# Use pip download instead (more robust, handles redirects & hashes)
packages = [
    "rasterio==1.5.2",
    "pyproj==3.8.0",
    "geopandas==1.0.1",
    "pyogrio==0.13.0",
    "affine==3.0.1",
    "attrs==26.1.0",
]

print("=== Downloading wheels via pip download ===")
for pkg in packages:
    print(f"\nDownloading: {pkg}")
    for attempt in range(1, 6):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "download",
             "--dest", "wheels",
             "--no-deps",
             "--timeout", "120",
             pkg],
            capture_output=True, text=True
        )
        print(result.stdout[-500:] if result.stdout else "")
        if result.returncode == 0:
            print(f"  OK: {pkg}")
            break
        else:
            print(f"  Attempt {attempt}/5 failed. Retrying in 5s...")
            print(result.stderr[-300:])
            time.sleep(5)

print("\n=== Installing from local wheels ===")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "--find-links", "wheels",
     "--no-index",
     "rasterio", "pyproj", "geopandas", "pyogrio", "affine", "attrs"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
print("Done.")
