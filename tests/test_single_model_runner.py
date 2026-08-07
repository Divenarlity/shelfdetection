from pathlib import Path
import re

from run_empty_shelf_test import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    find_unicode_font,
    select_labeled_test_image,
)


def test_default_model_is_empty_shelf_only():
    assert DEFAULT_MODEL.name == "empty_shelf_yolo11m_best.pt"


def test_deterministic_labeled_test_selection():
    image, label, count = select_labeled_test_image(DEFAULT_DATASET)
    assert image.name == "oos__db100_jpg.rf.dfdbdd41667b05ef3c326518a1d1b712.jpg"
    assert label.name == "oos__db100_jpg.rf.dfdbdd41667b05ef3c326518a1d1b712.txt"
    assert count == 2
    assert image.parent.name == "images"
    assert image.parent.parent.name == "test"


def test_unicode_font_can_measure_turkish_text():
    font = find_unicode_font()
    assert font.getbbox("Burada boşluk var")[2] > 0


def test_single_model_entry_remains_independent():
    runner_source = Path("run_empty_shelf_test.py").read_text(encoding="utf-8")
    assert re.search(r"""["']best[.]pt["']""", runner_source) is None
    assert "store_map" not in runner_source
    assert "ModelRunner" not in runner_source
