"""PROC ledger + pins: recorder, pipeline override, per-master persistence (v1.6.0)."""
import numpy as np

from lazystretch.io.pins import load_pins, save_pins
from lazystretch.objects.model import Parameters
from lazystretch.pipeline.ledger import DECIDED, INFO, MEASURED, Ledger
from lazystretch.pipeline.runcore import run_pipeline


def test_ledger_records_and_returns_computed():
    led = Ledger()
    assert led.record("Profile", "contrast", 0.12) == 0.12
    e = led.entries[0]
    assert e.key == "Profile|contrast" and e.computed == 0.12 and not e.pinned


def test_ledger_pin_overrides_and_type_coerces():
    led = Ledger({"Profile|contrast": "0.30"})       # string pin, float computed
    assert led.record("Profile", "contrast", 0.12) == 0.30
    assert led.entries[0].pinned is True


def test_ledger_info_and_measured_not_pinnable():
    led = Ledger({"Run|mode": "execute", "S|median": 0.9})
    assert led.record("Run", "mode", "preview", kind=INFO) == "preview"
    assert led.record("S", "median", 0.001, kind=MEASURED) == 0.001
    assert all(not e.pinned for e in led.entries)


def test_ledger_bad_pin_ignored():
    led = Ledger({"HDR|layers": "not-a-number"})
    assert led.record("HDR", "layers", 6) == 6          # coercion fails -> computed kept
    assert led.entries[0].pinned is False


def test_pipeline_returns_ledger_and_pin_takes_effect():
    img = np.clip(np.random.default_rng(0).random((64, 80, 3)) * 0.3, 0, 1)
    p = Parameters.for_object("emission")
    base = run_pipeline(img, p, preview=True)
    assert base.ledger is not None
    entry = next(e for e in base.ledger.entries if e.key == "Profile|contrast strength")
    assert not entry.pinned
    pinned = run_pipeline(img, p, preview=True, pins={"Profile|contrast strength": 0.0})
    e2 = next(e for e in pinned.ledger.entries if e.key == "Profile|contrast strength")
    assert e2.pinned and e2.value == 0.0


def test_pins_persist_per_master(tmp_path):
    master = tmp_path / "M42.fits"
    master.write_bytes(b"x")
    assert load_pins(str(master)) == {}
    save_pins(str(master), {"Profile|saturation boost": 0.7})
    assert load_pins(str(master)) == {"Profile|saturation boost": 0.7}
    save_pins(str(master), {})                          # empty removes the file
    assert not (tmp_path / "history" / "M42.pins.json").exists()
