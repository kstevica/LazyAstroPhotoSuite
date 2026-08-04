"""Run an external image tool via temp files: write input, invoke, read output.

Keeps the wall tools (StarNet, GraXpert) at arm's length — they get a plain image
file, run in their own process, and hand back a result file. Nothing about the tool
leaks into the core.
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from ..io.image_io import load_image, save_image


def run_image_roundtrip(
    cmd_builder: Callable[[str, str], Tuple[List[str], str]],
    img: np.ndarray,
    in_ext: str = ".fits",
    *,
    bit_depth: int = 16,
    timeout: int = 900,
) -> np.ndarray:
    """Write ``img`` to a temp file, run the tool, and return the output as an array.

    ``cmd_builder(in_path, tmp_dir)`` returns ``(argv, out_glob)`` — the command to run
    and a glob pattern that matches the tool's output file inside ``tmp_dir``. Raises
    ``RuntimeError`` if the tool fails or produces no output.
    """
    with tempfile.TemporaryDirectory(prefix="lazystretch_") as tmp:
        in_path = str(Path(tmp) / f"input{in_ext}")
        save_image(in_path, img, bit_depth=bit_depth)
        argv, out_glob = cmd_builder(in_path, tmp)
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
        except (subprocess.TimeoutExpired, OSError) as e:
            raise RuntimeError(f"{argv[0]} failed to run: {e}") from e
        matches = [m for m in glob.glob(out_glob) if os.path.abspath(m) != os.path.abspath(in_path)]
        if not matches:
            tail = (proc.stderr or proc.stdout or b"")[-400:].decode("utf-8", "replace")
            raise RuntimeError(
                f"{Path(argv[0]).name} produced no output (exit {proc.returncode}). {tail}")
        matches.sort(key=os.path.getmtime)
        return load_image(matches[-1]).data
