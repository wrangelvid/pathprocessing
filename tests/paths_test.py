import unittest
import numpy as np
from pathprocessing.paths import LinearPaths2D


class TestLinear2DPaths(unittest.TestCase):
    SQUARE = LinearPaths2D([np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])])

    def test_len(self):
        expected_value = 2
        paths = LinearPaths2D(
            [np.array([[0, 0], [1, 1]]), np.array([[0, 0], [0, 1], [1, 1]])]
        )
        result = len(paths)

        self.assertEqual(result, expected_value)

    def test_number_of_segments(self):
        expected_value = 3
        paths = LinearPaths2D(
            [np.array([[0, 0], [1, 1]]), np.array([[0, 0], [0, 1], [1, 1]])]
        )

        result = paths.number_of_segments

        self.assertEqual(result, expected_value)

    def test_bbox(self):
        expected_value = [-20, -4, 6, 3]
        paths = LinearPaths2D(
            [np.array([[-2, 3], [1, 1]]), np.array([[0, 0], [6, -4], [-20, 1]])]
        )

        result = paths.bbox

        self.assertSequenceEqual(result, expected_value)

    def test_tolist(self):
        expected_value = [[[-2, 3], [1, 1]], [[0, 0], [6, -4], [-20, 1]]]
        paths = LinearPaths2D(
            [np.array([[-2, 3], [1, 1]]), np.array([[0, 0], [6, -4], [-20, 1]])]
        )

        result = paths.tolist()

        self.assertSequenceEqual(result, expected_value)

    def test_compress(self):
        # Irrelevant intermediate values.
        expected_value = [[[0, 0], [0, 2]], [[0, 0], [2, 2]]]

        paths = LinearPaths2D(
            [np.array([[0, 0], [0, 1], [0, 2]]), np.array([[0, 0], [1, 1], [2, 2]])]
        )

        result = paths.compress().tolist()

        self.assertSequenceEqual(result, expected_value)

        # Rough resolution.
        expected_value = [[[0, 0], [0, 2]], [[0, 0], [2, 2]]]

        paths = LinearPaths2D(
            [
                np.array([[0, 0], [-0.3, 1], [0, 2]]),
                np.array([[0, 0], [0.5, 1], [2, 2]]),
            ]
        )

        result = paths.compress(resolution=1).tolist()

        self.assertSequenceEqual(result, expected_value)

    def test_shift(self):
        expected_value = [[[-3, 6], [0, 4]], [[-1, 3], [5, -1], [-21, 4]]]
        paths = LinearPaths2D(
            [np.array([[-2, 3], [1, 1]]), np.array([[0, 0], [6, -4], [-20, 1]])]
        )

        result = paths.shift(-1, 3).tolist()

        self.assertSequenceEqual(result, expected_value)

    def test_zero(self):
        expected_value = [[[18, 7], [21, 5]], [[20, 4], [26, 0], [0, 5]]]
        paths = LinearPaths2D(
            [np.array([[-2, 3], [1, 1]]), np.array([[0, 0], [6, -4], [-20, 1]])]
        )

        result = paths.zero().tolist()

        self.assertSequenceEqual(result, expected_value)

    def test_center(self):
        expected_value = [
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]]
        ]

        result = self.SQUARE.center().tolist()

        self.assertSequenceEqual(result, expected_value)

    def test_scale_to(self):
        shifted_square = self.SQUARE.shift(1, 1)

        # No scale.
        expected_value = [[[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]]
        result = shifted_square.scale_to().tolist()
        self.assertSequenceEqual(result, expected_value)

        # Scale width. Maintains aspect ratio.
        expected_value = [[[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]]
        result = shifted_square.scale_to(w=2).tolist()
        self.assertSequenceEqual(result, expected_value)

        # Scale height. Maintains aspect ratio.
        expected_value = [[[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]]
        result = shifted_square.scale_to(h=2).tolist()
        self.assertSequenceEqual(result, expected_value)

        # Scale width and height.
        expected_value = [[[1, 1], [3, 1], [3, 5], [1, 5], [1, 1]]]
        result = shifted_square.scale_to(w=2, h=4).tolist()
        self.assertSequenceEqual(result, expected_value)

        # Negative width and height scaling
        expected_value = [[[1, 1], [-1, 1], [-1, -3], [1, -3], [1, 1]]]
        result = shifted_square.scale_to(w=-2, h=-4).tolist()
        self.assertSequenceEqual(result, expected_value)

    def test_hflip(self):
        small_square = self.SQUARE.shift(-1)
        big_square = self.SQUARE.scale_to(2)
        paths = small_square + big_square

        expected_value = [
            [[2, 0], [1, 0], [1, 1], [2, 1], [2, 0]],
            [[1, 0], [-1, 0], [-1, 2], [1, 2], [1, 0]],
        ]
        result = paths.hflip().tolist()

        self.assertSequenceEqual(result, expected_value)

    def test_vflip(self):
        small_square = self.SQUARE.shift(-1)
        big_square = self.SQUARE.scale_to(2)
        paths = small_square + big_square

        expected_value = [
            [[-1, 2], [0, 2], [0, 1], [-1, 1], [-1, 2]],
            [[0, 2], [2, 2], [2, 0], [0, 0], [0, 2]],
        ]
        result = paths.vflip().tolist()

        self.assertSequenceEqual(result, expected_value)

    def test_rotate_by(self):
        expected_value = [[[1, 0], [1, 1], [0, 1], [0, 0], [1, 0]]]

        result = np.round(self.SQUARE.rotate_by(np.pi / 2).tolist()).tolist()

        self.assertSequenceEqual(result, expected_value)

    def test_slicing(self):
        paths = LinearPaths2D(
            [
                np.array([[0, 0], [1, 1]]),
                np.array([[0, 0], [0, 1], [1, 1]]),
                np.array([[3, -1], [2, 3], [1, 1]]),
            ]
        )

        # index element
        expected_value = [[0, 0], [1, 1]]
        result = paths[0].tolist()

        self.assertEqual(result, expected_value)

        # index last element
        expected_value = [[3, -1], [2, 3], [1, 1]]
        result = paths[-1].tolist()

        self.assertEqual(result, expected_value)

        # skip elements
        expected_value = [[[0, 0], [1, 1]], [[3, -1], [2, 3], [1, 1]]]
        result = paths[::2].tolist()

        self.assertEqual(result, expected_value)

    def test_minimum_length(self):
        paths = LinearPaths2D(
            [
                np.array([[0, 0], [0, 0.2]]),
                np.array([[0, 0], [0, 2], [1, 1]]),
                np.array([[3, -1], [2, 3], [1, 1]]),
            ]
        )

        expected_value = [[[0, 0], [0, 2], [1, 1]], [[3, -1], [2, 3], [1, 1]]]
        result = paths.minimum_length(1.0).tolist()

        self.assertEqual(result, expected_value)

    def test_unique(self):
        expected_value = [
            [[0, 0], [1, 1]],
            [[0, 0], [0, 1], [1, 1]],
            [[3, -1], [2, 3], [1, 1]],
        ]

        paths = LinearPaths2D(
            [
                np.array([[0, 0], [1, 1]]),
                np.array([[0, 0], [0, 1], [1, 1]]),
                np.array([[0, 0], [0, 1], [1, 1]]),
                np.array([[0, 0], [0, 1], [1, 1]]),
                np.array([[0, 0], [0, 1], [1, 1]]),
                np.array([[3, -1], [2, 3], [1, 1]]),
            ]
        )

        result = paths.unique().tolist()

        self.assertSequenceEqual(result, expected_value)

    def test_add(self):
        expected_value = [
            [[0, 0], [1, 1]],
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 1]],
            [[3, -1], [2, 3], [1, 1]],
        ]

        paths1 = LinearPaths2D(
            [np.array([[0, 0], [1, 1]]), np.array([[0, 0], [0, 1], [1, 1]])]
        )
        paths2 = LinearPaths2D(
            [np.array([[0, 0], [1, 1]]), np.array([[3, -1], [2, 3], [1, 1]])]
        )
        paths3 = paths1 + paths2

        result = paths3.tolist()

        self.assertSequenceEqual(result, expected_value)
