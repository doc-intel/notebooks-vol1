"""Output / cache root resolution — single source of truth for where the
package writes its persisted artefacts.

The package writes a lot : parsed `line_df.parquet`, `retrieved_pages`,
`answer.json`, the `storage.sqlite` shared DB, the V5 side-car
`_index.sqlite`, embeddings caches, LLM-call caches, the rendering cache,
the Streamlit uploads area, ... All of it used to anchor on the repo root
via `Path("output")` or `Path(__file__).resolve().parents[3] / "output"`,
so any consumer running the package with cwd=rag (the V6 Electron CLI
bridge, ad-hoc scripts, even some tests) would litter the repo with cache
files that have nothing to do with the manuscript.

The package is now neutral to deployment, the way `pandas.to_sql` is neutral
to which DB you point it at : it accepts an explicit destination, falls back
to an env var, and only as a last resort lands somewhere predictable on the
user's machine (NOT inside the source tree).

== Resolution order ==

`output_root()` resolves the cache root with this priority :

    1. Explicit argument (when the caller passes one)
    2. Environment variable ``DOCINTEL_OUTPUT_DIR``
       (alias ``DOCINTEL_HOME`` honoured for back-compat)
    3. Default : ``Path.home() / ".docintel" / "output"``

The default is **outside the repo**. The V6 Electron app sets
``DOCINTEL_OUTPUT_DIR`` to its ``userData`` directory so caches live with the
app. Tests set it to a tmp dir so they don't touch the dev's real cache.

== Public API ==

    >>> from lib.storage.paths import output_root, output_subdir
    >>> output_root()
    PosixPath('/home/alice/.docintel/output')
    >>> output_root(Path("/tmp/explicit"))
    PosixPath('/tmp/explicit')
    >>> output_subdir("rendering")
    PosixPath('/home/alice/.docintel/output/rendering')
"""
from __future__ import annotations

import os
from pathlib import Path


_ENV_VAR_PRIMARY = "DOCINTEL_OUTPUT_DIR"
_ENV_VAR_ALIASES: tuple[str, ...] = ("DOCINTEL_HOME",)


def _from_env() -> Path | None:
    """Return the env-var-provided root, or None when unset / empty."""
    for name in (_ENV_VAR_PRIMARY, *_ENV_VAR_ALIASES):
        v = os.environ.get(name)
        if v:
            return Path(v).expanduser()
    return None


def output_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the resolved cache root.

    Resolution order :

      1. ``explicit`` argument (caller-provided)
      2. ``$DOCINTEL_OUTPUT_DIR`` (alias ``$DOCINTEL_HOME``)
      3. ``Path.home() / ".docintel" / "output"`` (per-user, out of repo)

    The directory is NOT created here. Callers that write into it are
    responsible for ``mkdir(parents=True, exist_ok=True)`` on the eventual
    file's parent.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    env = _from_env()
    if env is not None:
        return env
    return Path.home() / ".docintel" / "output"


def output_subdir(
    name: str,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    create: bool = False,
) -> Path:
    """Return ``<output_root> / name``. Optionally creates it on disk.

    Thin convenience over ``output_root()`` for callers that always want a
    fixed sub-folder (e.g. ``"rendering"``, ``"_streamlit_uploads"``).
    """
    p = output_root(output_dir) / name
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    "output_root",
    "output_subdir",
]
