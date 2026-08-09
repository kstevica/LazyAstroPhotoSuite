"""Milky Way (widefield) class — profile, global gating, desat=0 rule (v1.6.1)."""
import numpy as np

from lazystretch.data.loader import get_data
from lazystretch.objects.model import Parameters
from lazystretch.processes import finishing


def test_milkyway_profile_present_and_values():
    data = get_data()
    assert "milkyway" in data.class_list
    p = data.profile_for("milkyway")
    assert p.bgLevel == 0.30 and p.clip == -0.95 and p.sat == 0.50
    assert p.contrast == 0.20 and p.starReduce is False and p.starLevel == 1.00
    assert p.hdr is True and p.scnr is True and p.localContrast is True


def test_milkyway_profile_drives_toggles():
    p = Parameters.for_object("milkyway")
    assert p.object_class == "milkyway"
    assert p.doStarReduce is False        # the star carpet IS the subject
    assert p.doHDR is True and p.doSCNR is True and p.doLocalContrast is True


def test_milkyway_uses_global_behaviour():
    data = get_data()
    # deliberately absent from every class predicate -> global (not masked) behaviour
    for pred in ("adaptiveFloorAppliesTo", "maskedDarkenAppliesTo", "starProtectAppliesTo"):
        assert "milkyway" not in data.class_predicates[pred]


def _warm(shape=(64, 64)):
    base = np.full(shape, 0.4)
    img = np.stack([base * 1.0, base * 0.85, base * 0.7], axis=-1)  # warm R>G>B tint
    return np.clip(img, 0, 1)


def test_milkyway_skips_tint_desaturation():
    img = _warm()
    generic = finishing.dehaze(img.copy(), 0.6, 0.1, "generic")
    milky = finishing.dehaze(img.copy(), 0.6, 0.1, "milkyway")

    def chroma(a):
        return float(np.mean(a[..., 0] - a[..., 2]))   # R-B spread = warm saturation

    # generic desaturates the warm tint; milkyway keeps it (dose 0)
    assert chroma(milky) > chroma(generic) + 1e-4
    assert chroma(milky) >= chroma(img) - 1e-6           # milkyway does not grey it
