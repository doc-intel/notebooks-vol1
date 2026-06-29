"""EasyOCR Reader factory.

EasyOCR ships pretrained text-detection + recognition models that load lazily
on first call. The first `Reader(...)` invocation downloads weights to a local
cache ; subsequent calls are offline. We cache the instance per (languages,
gpu) tuple so a corpus-wide run does not pay the load cost per PDF.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def get_easyocr_reader(
    languages: tuple[str, ...] = ("en",),
    gpu: bool = False,
):
    """Return a cached `easyocr.Reader` for the given language set.

    Languages follow EasyOCR's ISO-639-1 codes: `en`, `fr`, `de`, `es`, ... See
    https://www.jaided.ai/easyocr/ for the full list. The first call per
    (languages, gpu) tuple downloads model weights (~150 MB) ; every call after
    that is offline.

    `gpu=True` requires a CUDA-capable PyTorch install ; default is CPU-only
    so the module stays usable on any machine.
    """
    import easyocr  # imported lazily so a missing easyocr only fails on use
    # verbose=False suppresses the progress bar that uses unicode block chars
    # easyocr fails to render on Windows cp1252 stdout.
    return easyocr.Reader(list(languages), gpu=gpu, verbose=False)
