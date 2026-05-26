"""Shared download-progress helpers for OpenBlender components."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("openblender3d.download")


def _comfy_tqdm_class():
    """Return a tqdm class that mirrors byte progress to ComfyUI ProgressBar."""
    try:
        import tqdm as _tqdm_mod
        import comfy.utils as _cu
    except Exception:
        return None

    class _ComfyTqdm(_tqdm_mod.tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            kwargs.setdefault("mininterval", 0.2)
            kwargs.setdefault("maxinterval", 2.0)
            super().__init__(*args, **kwargs)
            self._comfy_total = int(self.total or 0)
            self._comfy_done = 0
            self._comfy_pbar = _cu.ProgressBar(self._comfy_total) if self._comfy_total > 0 else None

        def update(self, n=1):
            ret = super().update(n)
            if self._comfy_pbar and n:
                self._comfy_done = min(self._comfy_done + int(n), self._comfy_total)
                self._comfy_pbar.update_absolute(self._comfy_done, self._comfy_total)
            return ret

    return _ComfyTqdm


def install_hf_progress_patch() -> None:
    """Ensure HF downloads always show progress in terminal + Comfy progress UI."""
    try:
        from huggingface_hub.utils import enable_progress_bars
        import huggingface_hub as hfh
    except Exception as exc:
        log.debug("huggingface_hub patch skipped: %s", exc)
        return

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    try:
        enable_progress_bars()
    except Exception:
        pass

    if getattr(hfh, "_openblender_progress_patched", False):
        return

    tqdm_cls = _comfy_tqdm_class()
    if tqdm_cls is None:
        return

    orig_hf_hub_download = hfh.hf_hub_download
    orig_snapshot_download = hfh.snapshot_download

    def _hf_hub_download(*args, **kwargs):
        kwargs.setdefault("tqdm_class", tqdm_cls)
        try:
            return orig_hf_hub_download(*args, **kwargs)
        except TypeError:
            kwargs.pop("tqdm_class", None)
            return orig_hf_hub_download(*args, **kwargs)

    def _snapshot_download(*args, **kwargs):
        kwargs.setdefault("tqdm_class", tqdm_cls)
        try:
            return orig_snapshot_download(*args, **kwargs)
        except TypeError:
            kwargs.pop("tqdm_class", None)
            return orig_snapshot_download(*args, **kwargs)

    hfh.hf_hub_download = _hf_hub_download
    hfh.snapshot_download = _snapshot_download
    hfh._openblender_progress_patched = True
    log.info("[download] HuggingFace progress bars enabled")


def download_url_with_progress(url: str, destination: str | Path, desc: str | None = None) -> Path:
    """Download an HTTP URL with byte progress shown in ComfyUI console."""
    import requests

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")

    name = desc or target.name
    tqdm_cls = _comfy_tqdm_class()

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0) or 0)
        with open(temp, "wb") as fout:
            if tqdm_cls and total > 0:
                with tqdm_cls(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=name) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        fout.write(chunk)
                        bar.update(len(chunk))
            else:
                done = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fout.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        pct = (done * 100.0) / total
                        print(f"[download] {name}: {pct:5.1f}% ({done}/{total} bytes)", file=sys.stderr, flush=True)

    temp.replace(target)
    return target
