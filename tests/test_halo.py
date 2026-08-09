"""Halo Tamer — reflection filter-ring detection + per-channel subtraction (v1.5.1)."""
import numpy as np
import pytest

from lazystretch.data.loader import get_data
from lazystretch.objects.model import Parameters
from lazystretch.processes import halo

pytest.importorskip("photutils")


_SIZE = 1200
_C = 600         # frame centre; RMAX = 1200//8 = 150, so a 100px ring is in-window


def _ringed_star(size=_SIZE, cx=_C, cy=_C, ring_r=100, ring_amp=0.05, star_amp=0.9):
    """A dark sky with one bright star and a faint concentric reflection ring."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = np.full((size, size), 0.02)
    img += star_amp * np.exp(-(r / 3.0) ** 2)                     # the star (tight)
    img += ring_amp * np.exp(-((r - ring_r) / 13.0) ** 2)        # the ring (broad annulus)
    return np.clip(img, 0, 1)


def test_halo_scan_finds_the_ring():
    rings = halo.halo_scan(_ringed_star())
    assert len(rings) >= 1
    g = rings[0]
    assert 70 < g["rp"] < 130 and g["amp"] > 0.012      # near the injected 100px ring


def test_halo_tamer_reduces_ring_amplitude():
    img = _ringed_star(ring_amp=0.06)
    out = halo.halo_tamer(img)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    r = np.sqrt((xx - _C) ** 2 + (yy - _C) ** 2)
    band = (r > 88) & (r < 112)
    assert np.median(out[band]) < np.median(img[band])   # ring suppressed
    core = (r < 6)
    assert np.allclose(out[core], img[core])              # star core guarded


def test_halo_scan_none_on_plain_field():
    plain = np.clip(np.random.default_rng(0).normal(0.05, 0.005, (900, 900)), 0, 1)
    assert halo.halo_scan(plain) == []


def test_halo_scan_small_frame_is_noop():
    assert halo.halo_scan(np.zeros((200, 200))) == []    # RMAX < 90 -> skip


def test_halo_tamer_per_channel_amps():
    # red-dominant ring (like an OSC filter ghost)
    base = _ringed_star(ring_amp=0.0)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE].astype(np.float64)
    r = np.sqrt((xx - _C) ** 2 + (yy - _C) ** 2)
    ring = np.exp(-((r - 100) / 13.0) ** 2)
    rgb = np.stack([base + 0.08 * ring, base + 0.04 * ring, base + 0.02 * ring], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    rings = halo.halo_scan(rgb)
    assert len(rings) >= 1
    amps = rings[0]["amps"]
    assert amps[0] > amps[1] > amps[2]                    # per-channel: R > G > B


def test_milkyway_profile_disables_halo_tamer():
    data = get_data()
    assert data.profile_for("milkyway").haloTamer is False
    assert data.profile_for("emission").haloTamer is True
    assert Parameters.for_object("milkyway").haloTamer is False
    assert Parameters.for_object("emission").haloTamer is True
