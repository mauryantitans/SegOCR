"""Tests for segocr.postprocessing.cleanup."""
from __future__ import annotations

import numpy as np

from segocr.postprocessing.cleanup import cleanup_prediction


def test_confidence_threshold_zeroes_low_conf_regions():
    # Two 8x8 blobs, one high-confidence (class 1) and one low (class 2).
    pred = np.zeros((20, 20), dtype=np.int32)
    pred[2:10, 2:10] = 1   # high-confidence blob
    pred[2:10, 12:20] = 2  # low-confidence blob — should be removed
    conf = np.zeros_like(pred, dtype=np.float32)
    conf[2:10, 2:10] = 0.9
    conf[2:10, 12:20] = 0.1
    out = cleanup_prediction(pred, conf, threshold=0.5, min_component_area=4)
    assert out[5, 5] == 1     # class-1 blob center survives
    assert (out == 2).sum() == 0   # all class-2 pixels gone


def test_area_filter_drops_small_components():
    pred = np.zeros((30, 30), dtype=np.int32)
    # A 2x2 blob (area 4) — should be dropped at min_component_area=20
    pred[0:2, 0:2] = 1
    # A 6x6 blob (area 36) — should survive
    pred[10:16, 10:16] = 1
    conf = np.ones_like(pred, dtype=np.float32)
    out = cleanup_prediction(pred, conf, threshold=0.5, min_component_area=20)
    assert out[1, 1] == 0
    assert out[12, 12] == 1


def test_area_filter_drops_single_pixel_speckle():
    # A solid blob plus a single-pixel speckle. The single pixel may or
    # may not survive morphology on cv2's particular kernel, but the
    # area filter (min_component_area=20) guarantees it's gone.
    pred = np.zeros((40, 40), dtype=np.int32)
    pred[5, 5] = 1                 # 1-pixel speckle
    pred[15:25, 15:25] = 1         # 10x10 solid blob
    conf = np.ones_like(pred, dtype=np.float32)
    out = cleanup_prediction(pred, conf, threshold=0.5, min_component_area=20)
    # speckle gone, solid blob kept
    assert out[5, 5] == 0
    assert out[20, 20] == 1


def test_all_background_returns_all_zeros():
    pred = np.zeros((10, 10), dtype=np.int32)
    conf = np.ones_like(pred, dtype=np.float32)
    out = cleanup_prediction(pred, conf, threshold=0.5)
    assert out.sum() == 0


def test_shape_mismatch_raises():
    pred = np.zeros((10, 10), dtype=np.int32)
    conf = np.ones((10, 11), dtype=np.float32)
    import pytest
    with pytest.raises(ValueError):
        cleanup_prediction(pred, conf)
