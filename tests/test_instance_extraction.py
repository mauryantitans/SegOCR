"""Tests for segocr.postprocessing.instance_extraction."""
from __future__ import annotations

import numpy as np

from segocr.postprocessing.instance_extraction import (
    CharacterInstance,
    extract_instances,
)


def test_two_separate_blobs_produce_two_instances():
    clean = np.zeros((40, 40), dtype=np.int32)
    clean[5:15, 5:15] = 1     # blob A
    clean[5:15, 25:35] = 1    # blob B
    instances = extract_instances(clean, min_size=4, max_size=40)
    assert len(instances) == 2
    assert all(inst.class_id == 1 for inst in instances)


def test_different_classes_produce_separate_instances():
    clean = np.zeros((40, 40), dtype=np.int32)
    clean[5:15, 5:15] = 1     # class 1
    clean[5:15, 25:35] = 2    # class 2
    instances = extract_instances(clean, min_size=4, max_size=40)
    classes = sorted(inst.class_id for inst in instances)
    assert classes == [1, 2]


def test_min_size_filter():
    clean = np.zeros((40, 40), dtype=np.int32)
    clean[10:12, 10:12] = 1   # 2x2 — below min_size=4
    clean[20:28, 20:28] = 2   # 8x8 — above min_size
    instances = extract_instances(clean, min_size=4, max_size=40)
    assert len(instances) == 1
    assert instances[0].class_id == 2


def test_max_size_filter_rejects_huge_blobs():
    clean = np.zeros((100, 100), dtype=np.int32)
    clean[5:95, 5:95] = 1   # 90x90 — above max_size=50
    instances = extract_instances(clean, min_size=4, max_size=50)
    assert len(instances) == 0


def test_empty_map_returns_empty_list():
    clean = np.zeros((20, 20), dtype=np.int32)
    assert extract_instances(clean) == []


def test_instance_has_correct_bbox_and_centroid():
    clean = np.zeros((40, 40), dtype=np.int32)
    clean[10:20, 5:15] = 1   # bbox (5, 10, 10, 10), centroid ~(9.5, 14.5)
    instances = extract_instances(clean, min_size=4, max_size=40)
    assert len(instances) == 1
    inst = instances[0]
    assert isinstance(inst, CharacterInstance)
    assert inst.bbox == (5, 10, 10, 10)
    assert inst.area == 100
    # cv2 returns the centroid of pixel centers, which for an integer grid
    # on a 10x10 block starting at (5,10) is roughly (9.5, 14.5).
    assert abs(inst.centroid[0] - 9.5) < 1.0
    assert abs(inst.centroid[1] - 14.5) < 1.0
