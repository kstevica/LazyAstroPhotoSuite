"""Disk-backed per-master run history (io/history.py)."""
import numpy as np

from lazystretch.io.history import (
    MAX_ITEMS,
    STATUS_DISCARDED,
    STATUS_SELECTED,
    HistoryStore,
)


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.random((40, 50, 3)), 0, 1)


def _master(tmp_path):
    m = tmp_path / "M42.tif"
    m.write_bytes(b"x")                 # only the path matters to the store
    return str(m)


def test_add_writes_image_and_index(tmp_path):
    s = HistoryStore(_master(tmp_path))
    it = s.add(_img(), {"objectClass": "emission", "highlights": 0.05}, "run A", "preview", 17)
    assert (s.dir / it["file"]).exists()               # PNG on disk
    assert s.index_path.exists()                        # JSON index on disk
    assert s.load_image_data(it).shape[:2] == (40, 50)  # loads back


def test_persists_across_reopen(tmp_path):
    master = _master(tmp_path)
    s = HistoryStore(master)
    s.add(_img(1), {"objectClass": "emission"}, "A", "preview", 10)
    s.add(_img(2), {"objectClass": "emission"}, "B", "preview", 11)
    s2 = HistoryStore(master)                           # reopen -> reload from disk
    assert len(s2.items) == 2
    assert s2.items[0]["label"] == "A" and s2.items[1]["label"] == "B"


def test_capacity_evicts_oldest_unrated_first(tmp_path):
    s = HistoryStore(_master(tmp_path))
    first = s.add(_img(), {}, "keepme", "preview", 1)
    s.set_rating(first, 5)                              # rated -> protected
    files = [first["file"]]
    for k in range(MAX_ITEMS + 3):                      # overflow
        files.append(s.add(_img(k), {}, f"r{k}", "preview", 1)["file"])
    assert len(s.items) == MAX_ITEMS
    assert first in s.items                             # the rated one survived
    assert (s.dir / first["file"]).exists()
    # an evicted (unrated, early) item's file is gone
    assert not (s.dir / files[1]).exists()


def test_rating_comment_status_persist(tmp_path):
    master = _master(tmp_path)
    s = HistoryStore(master)
    it = s.add(_img(), {}, "A", "preview", 1)
    s.set_rating(it, 4)
    s.set_comment(it, "nice core")
    s.set_status(it, STATUS_SELECTED)
    s2 = HistoryStore(master)
    got = s2.items[0]
    assert got["rating"] == 4 and got["comment"] == "nice core" and got["status"] == STATUS_SELECTED


def test_remove_and_clear_delete_files(tmp_path):
    s = HistoryStore(_master(tmp_path))
    a = s.add(_img(1), {}, "A", "preview", 1)
    b = s.add(_img(2), {}, "B", "preview", 1)
    s.remove(a)
    assert not (s.dir / a["file"]).exists() and len(s.items) == 1
    s.clear()
    assert s.items == [] and not (s.dir / b["file"]).exists()


def test_rating_clamped_and_status_validated(tmp_path):
    s = HistoryStore(_master(tmp_path))
    it = s.add(_img(), {}, "A", "preview", 1)
    s.set_rating(it, 9)
    assert it["rating"] == 5
    s.set_rating(it, -3)
    assert it["rating"] == 0
    s.set_status(it, "bogus")                           # ignored
    assert it["status"] != "bogus"
