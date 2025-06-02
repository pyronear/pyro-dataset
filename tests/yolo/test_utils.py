"""
Tests for the YOLO utility functions.
"""

import numpy as np

from pyro_dataset.yolo.utils import xywhn2xyxyn, xyxyn2xywhn


def test_xywhn2xyxyn_single_box():
    """Test conversion of a single bounding box from xywhn to xyxyn format."""
    # Arrange: center_x, center_y, width, height
    bbox_xywhn = np.array([0.5, 0.5, 0.2, 0.4], dtype=np.float16)

    # Act: convert to xyxyn format (top-left x, top-left y, bottom-right x, bottom-right y)
    bbox_xyxyn = xywhn2xyxyn(bbox_xywhn)

    # Assert: verify the conversion is correct
    expected = np.array([0.4, 0.3, 0.6, 0.7], dtype=np.float16)
    np.testing.assert_almost_equal(bbox_xyxyn, expected, decimal=5)


def test_xywhn2xyxyn_multiple_boxes():
    """Test conversion of multiple bounding boxes from xywhn to xyxyn format."""
    # Arrange: multiple bounding boxes in xywhn format
    bboxes_xywhn = np.array(
        [
            [0.5, 0.5, 0.2, 0.4],  # center box
            [0.1, 0.1, 0.1, 0.1],  # top-left box
            [0.9, 0.9, 0.1, 0.1],  # bottom-right box
        ],
        dtype=np.float16,
    )

    # Act: convert to xyxyn format
    bboxes_xyxyn = xywhn2xyxyn(bboxes_xywhn)

    # Assert: verify the conversion is correct for all boxes
    expected = np.array(
        [
            [0.4, 0.3, 0.6, 0.7],  # center box in xyxyn
            [0.05, 0.05, 0.15, 0.15],  # top-left box in xyxyn
            [0.85, 0.85, 0.95, 0.95],  # bottom-right box in xyxyn
        ],
        dtype=np.float16,
    )
    np.testing.assert_almost_equal(bboxes_xyxyn, expected, decimal=3)


def test_xywhn2xyxyn_edge_cases():
    """Test conversion of edge case bounding boxes from xywhn to xyxyn format."""
    # Arrange: edge case bounding boxes
    bboxes_xywhn = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # zero size box at origin
            [1.0, 1.0, 0.0, 0.0],  # zero size box at bottom-right
            [0.5, 0.5, 1.0, 1.0],  # full image box
        ],
        dtype=np.float16,
    )

    # Act: convert to xyxyn format
    bboxes_xyxyn = xywhn2xyxyn(bboxes_xywhn)

    # Assert: verify the conversion is correct for edge cases
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # zero size box at origin in xyxyn
            [1.0, 1.0, 1.0, 1.0],  # zero size box at bottom-right in xyxyn
            [0.0, 0.0, 1.0, 1.0],  # full image box in xyxyn
        ],
        dtype=np.float16,
    )
    np.testing.assert_almost_equal(bboxes_xyxyn, expected, decimal=5)


def test_xywhn2xyxyn_data_type_preservation():
    """Test that the function preserves the data type."""
    # Arrange: bounding box with float16 type
    bbox_xywhn = np.array([0.5, 0.5, 0.2, 0.4], dtype=np.float16)

    # Act: convert to xyxyn format
    bbox_xyxyn = xywhn2xyxyn(bbox_xywhn)

    # Assert: verify the output type is float (as specified in the function)
    assert (
        bbox_xyxyn.dtype == np.float16
    ), "Output type should be float as specified in the function"


def test_xywhn2xyxyn_inverse_operation():
    """Test that xywhn2xyxyn and xyxyn2xywhn are inverse operations."""
    # Arrange: original bounding box in xywhn format
    original_bbox = np.array([0.5, 0.5, 0.2, 0.4], dtype=np.float16)

    # Act: convert to xyxyn and back to xywhn
    bbox_xyxyn = xywhn2xyxyn(original_bbox)
    bbox_xywhn = xyxyn2xywhn(bbox_xyxyn)

    # Assert: verify we get back the original values
    np.testing.assert_almost_equal(bbox_xywhn, original_bbox, decimal=3)


def test_xywhn2xyxyn_batch_dimension():
    """Test that the function works with batch dimensions."""
    # Arrange: batch of bounding boxes with an extra dimension
    batch_bboxes = np.array(
        [
            [[0.5, 0.5, 0.2, 0.4]],
            [[0.1, 0.1, 0.1, 0.1]],
        ],
        dtype=np.float16,
    )

    # Act: convert to xyxyn format
    batch_xyxyn = xywhn2xyxyn(batch_bboxes)

    # Assert: verify the conversion is correct and dimensions are preserved
    expected = np.array(
        [
            [[0.4, 0.3, 0.6, 0.7]],
            [[0.05, 0.05, 0.15, 0.15]],
        ],
        dtype=np.float16,
    )
    np.testing.assert_almost_equal(batch_xyxyn, expected, decimal=4)
    assert (
        batch_xyxyn.shape == batch_bboxes.shape
    ), "Output shape should match input shape"
