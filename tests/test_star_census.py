"""Star census (photutils) + the XISF-write guard — P1 close-out minor fixes."""
import numpy as np
import pytest

from lazystretch.io.image_io import save_image
from lazystretch.objects.analyze import analyze_view, star_census
from lazystretch.objects.model import Parameters

photutils = pytest.importorskip("photutils")   # census needs the optional dep


def _star_field(H=240, W=320, n=40, seed=0):
    """A dark sky with n bright gaussian stars + faint noise."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(0.02, 0.004, (H, W)), 0, 1)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n):
        cy, cx = rng.integers(10, H - 10), rng.integers(10, W - 10)
        amp = rng.uniform(0.4, 0.95)
        img += amp * np.exp(-(((xx - cx) / 1.6) ** 2 + ((yy - cy) / 1.6) ** 2))
    return np.clip(img, 0, 1)


def test_star_census_detects_and_summarizes():
    sc = star_census(_star_field(n=40))
    assert sc is not None
    assert sc["count"] > 10                       # finds a good fraction of the 40
    assert sc["perMpx"] > 0 and sc["medDiam"] > 0
    assert 0 <= sc["saturated"] <= sc["satChecked"] <= sc["count"]


def test_star_census_flat_field_is_none():
    flat = np.full((120, 160), 0.05)              # no stars -> graceful None
    assert star_census(flat) is None


def test_star_census_rgb_supported():
    rgb = np.stack([_star_field(seed=1)] * 3, axis=-1)
    sc = star_census(rgb)
    assert sc is not None and sc["count"] > 10


def test_analyze_view_reports_stars():
    img = np.stack([_star_field(n=30)] * 3, axis=-1)
    res = analyze_view(img, Parameters.for_object("open"))
    assert any(line.startswith("Stars:") for line in res.lines)


def test_xisf_write_is_refused(tmp_path):
    with pytest.raises(ValueError, match="XISF write is not supported"):
        save_image(str(tmp_path / "out.xisf"), np.zeros((8, 8, 3)))
