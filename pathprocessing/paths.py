import numpy as np
from rdp import rdp
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
import cairo
import qrcode
import os

import numpy.typing as npt
from typing import Union


class LinearPaths2D:
    """A class for a collection of linear paths in 2D.

    Attributes:
        bbox: Minimum bounding box xyxy style.
        width: Width (x-axis) of paths.
        height: Height (y-axis) of paths.
        number_of_segments: Total number of segments.
            A path with N points will have N - 1 segments.
    """

    def __init__(self, paths: list[npt.NDArray[np.float64]] = []):
        """Inits LinearPaths2D with list of paths.

        A path is a list of points stored as numpy arrays.

        Args:
            paths: Each path can have arbitrary points with the shape (N, 2).

        Raises:
            ValueError: An error is raised if the path shapes are wrong.
        """
        if paths:
            if not all([len(path.shape) == 2 for path in paths]):
                raise ValueError("All paths must be two dimensional arrays.")

            if not all([path.shape[1] == 2 for path in paths]):
                raise ValueError("All paths must be two dimensional.")

        self._paths = [path.astype(np.float64) for path in paths]

        # Get bounding box.
        all_points = np.vstack(self._paths)
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)
        self.bbox = [min_x, min_y, max_x, max_y]

        self.width = float(abs(max_x - min_x))
        self.height = float(abs(max_y - min_y))

        # Get number of segments.
        self.number_of_segments = sum([len(path) - 1 for path in self._paths])

    def viz(self, color: str = None) -> None:
        """Visualize paths.

        Args:
            color: If not set, will color every path differently.
                Use standard matplotlib color names.
        """
        for path in self._paths:
            plt.plot(path[:, 0], path[:, 1], color = color)

        plt.axis("equal")

    def tolist(self) -> list[list]:
        """Returns a list of paths.

        Converts each path from its numpy representation to a list.

        Returns:
            A list of list representing the set of paths.
        """
        return [path.tolist() for path in self._paths]

    def compress(self, resolution: float = 1e-3) -> "LinearPaths2D":
        """Simplifies paths to given resolution.

        Uses the Ramer-Douglas-Peucker algorithm to reduce
        the number of points in a curve by approximating it to
        a given resolution.

        Args:
            resolution: The epsilon in the rdp algorithm.

        Returns:
            A LinerPaths2D object with approximated curves.
        """
        return LinearPaths2D([rdp(path, resolution) for path in self._paths])

    def shift(self, x: float = 0.0, y: float = 0.0) -> "LinearPaths2D":
        """Shifts all points by given offsets.

        Args:
            x: Shift offset in x axis.
            y: Shift offset in y axis.

        Returns:
            A shifted LinearPaths2D object.
        """
        return LinearPaths2D([path + np.array([x, y]) for path in self._paths])

    def zero(self) -> "LinearPaths2D":
        """Shifts all points, such that minimum is at (0, 0)

        Returns:
            A zeroed LinearPaths2D object, with the minimum at 0, 0.
        """
        min_x, min_y, _, _ = self.bbox
        return LinearPaths2D([path - np.array([min_x, min_y]) for path in self._paths])

    def center(self) -> "LinearPaths2D":
        """Centers paths.

        Shifts all points such that bounding center is (0,0).

        Returns:
            A centered LinearPaths2D object.
        """
        min_x, min_y, _, _ = self.bbox

        return self.shift(x=-min_x - self.width / 2, y=-min_y - self.height / 2)

    def scale_to(self, w: float = None, h: float = None) -> "LinearPaths2D":
        """Scales Paths to a desired dimensions.

        Scaling is with respect to the minimum corner.
        The paths will be zeroed, scaled and then shifted back.

        Maintains aspect ratio if only one of the dimensions is given.

        Args:
            w: Desired width of the set of paths.
            h: Desired height of the set of paths.

        Returns:
            A scaled LinearPaths2D object.
        """
        min_x, min_y, _, _ = self.bbox

        if w is None and h is None:
            # Don't scale.
            scaling_factor_h = 1.0
            scaling_factor_w = 1.0
        elif w is None:
            scaling_factor_h = h / self.height
            scaling_factor_w = scaling_factor_h
        elif h is None:
            scaling_factor_w = w / self.width
            scaling_factor_h = scaling_factor_w
        else:
            scaling_factor_h = h / self.height
            scaling_factor_w = w / self.width

        scaled_paths = []
        for path in self._paths:
            # Shift to zero.
            shifted_path = path - np.array([min_x, min_y])
            # Scale.
            shifted_path[:, 0] *= scaling_factor_w
            shifted_path[:, 1] *= scaling_factor_h
            # Shift back.
            shifted_path += np.array([min_x, min_y])

            scaled_paths.append(shifted_path)

        return LinearPaths2D(scaled_paths)

    def hflip(self) -> "LinearPaths2D":
        """Flips paths along horizontal centerline.

        Returns:
            A horizontally flipped LinerPaths2D object.

        """
        return self.scale_to(w=-self.width, h=self.height).shift(x=self.width)

    def vflip(self) -> "LinearPaths2D":
        """Flips paths along vertical centerline.

        Returns:
            A vertically flipped LinerPaths2D object.

        """
        return self.scale_to(w=self.width, h=-self.height).shift(y=self.height)

    def rotate_by(self, theta) -> "LinearPaths2D":
        """Rotate by given angle with respect to center.

        Args:
            theta: Desired rotation in radians (CCW).

        Returns:
            A rotated  LinearPaths2D object.
        """
        # 2D Rotation matrix.
        # Note, this is the inverse of the common matrix.
        # This is because we rotate the points by multiplying
        # points @ R_inv, rather then the usual convention.
        R_inv = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

        min_x, min_y, _, _ = self.bbox

        centered_paths = self.shift(
            x=-min_x - self.width / 2, y=-min_y - self.height / 2
        )

        rotated_paths = []
        for paths in centered_paths:
            rotated_paths.append(paths.dot(R_inv))

        return LinearPaths2D(rotated_paths).shift(
            min_x + self.width / 2, min_y + self.height / 2
        )

    def minimum_length(self, minimum_length: float = 0.0) -> "LinearPaths2D":
        """Filters paths by minimum length.

        Args:
            minimum_length: Minimum acceptable path length (inclusive).

        Returns:
            A LinearPaths2D object with all paths at least
            as long as the minimum_length.
        """

        def path_length(path):
            pairwise_distance = np.sqrt(
                np.sum(np.square(path[1:] - path[:-1]), axis=1)
            )
            return np.sum(pairwise_distance)

        return LinearPaths2D(
            list(filter(lambda path: path_length(path) >= minimum_length, self._paths))
        )

    def unique(self) -> "LinearPaths2D":
        """Prunes duplicate paths.

        Returns:
            A LinerPaths2D object with unique paths.
        """

        paths_tuple = [tuple((tuple(coord) for coord in path)) for path in self._paths]
        return LinearPaths2D([np.array(path) for path in set(paths_tuple)])

    def __getitem__(self, i) -> Union[npt.NDArray[np.float64], "LinearPaths2D"]:
        if isinstance(i, int):
            return self._paths[i]
        return LinearPaths2D(self._paths[i])

    def __len__(self) -> int:
        """Returns number of paths.

        Returns:
            Number of paths.
        """
        return len(self._paths)

    def __add__(self, other) -> "LinearPaths2D":
        """Adds the paths of two objects.

        Doesn't take account for duplicates!

        Returns:
            A LinearPaths2D object with the paths from both objects.
        """
        return LinearPaths2D(self._paths + other._paths)

    @staticmethod
    def hstack(
        list_of_paths: list["LinearPaths2D"], offset: float = 0.0
    ) -> "LinearPaths2D":
        """Stacks paths objects horizontally.

        Does not zero paths before stacking.
        Stacks from left to right in the positive x direction.

        Args:
            list_of_paths: Objects to stack.
            offset: The spacing between the paths objects.

        Returns:
            A horizontally stacked LinerPaths2D object.
        """
        paths = list_of_paths[0][:]
        _, _, max_x, _ = paths.bbox
        for next_paths in list_of_paths[1:]:
            paths += next_paths.shift(max_x + offset)
            _, _, max_x, _ = paths.bbox

        return paths

    @staticmethod
    def vstack(
        list_of_paths: list["LinearPaths2D"], offset: float = 0.0
    ) -> "LinearPaths2D":
        """Stacks paths objects vertically.

        Does not zero paths before stacking.
        Stacks from left to right in the positive y direction.

        Args:
            list_of_paths: Objects to stack.
            offset: The spacing between the paths objects.

        Returns:
            A vertically stacked LinerPaths2D object.
        """
        paths = list_of_paths[0][:]
        _, _, _, max_y = paths.bbox
        for next_paths in list_of_paths[1:]:
            paths += next_paths.shift(y=max_y + offset)
            _, _, _, max_y = paths.bbox

        return paths

    @staticmethod
    def from_svg(file_name: str) -> "LinearPaths2D":
        """Reads vector paths from an SVG.

        Approximates Bezier curve, arcs, etc. by taking
        many points along the contour.

        Args:
            file_name: Absolute path to the SVG file.

        Returns:
            A LinearPaths2D object with the paths.
        """
        svg_paths, _ = svg2paths(file_name)
        paths = []
        # Need to find the continues sub paths.
        for path in sum([path.continuous_subpaths() for path in svg_paths], []):
            if path:
                # Need to linearize the path.
                linear_path = np.concatenate(
                    [
                        segment.point(np.linspace(0, 1, int(np.ceil(segment.length()))))
                        for segment in path
                    ]
                )
                paths += [np.vstack([linear_path.real, linear_path.imag]).T]

        return LinearPaths2D(paths).vflip()

    @staticmethod
    def from_string(
        text: str, font: str = "Poddins", slant: str = "NORMAL", weight: str = "NORMAL"
    ) -> "LinearPaths2D":
        """Creates a LinearPaths2D object from text.

        Args:
            text: Desired text.
            font: Any of the supported cairo fonts.
            slant: Choose from NORMAL, ITALIC, OBLIQUE.
            weight: Choose from NORMAL, BOLD.
        """
        _FONT_SIZE = 50
        _TEMP_FILE_NAME = "tmp_pathprocessing"

        slant_dict = {
            "NORMAL": cairo.FONT_SLANT_NORMAL,
            "ITALIC": cairo.FONT_SLANT_ITALIC,
            "OBLIQUE": cairo.FONT_SLANT_OBLIQUE,
        }

        weight_dict = {
            "NORMAL": cairo.FONT_WEIGHT_NORMAL,
            "BOLD": cairo.FONT_WEIGHT_BOLD,
        }

        with cairo.SVGSurface(
            _TEMP_FILE_NAME, len(text) * _FONT_SIZE, _FONT_SIZE + 2
        ) as surface:
            Context = cairo.Context(surface)
            Context.set_font_size(_FONT_SIZE)

            # Font Style
            Context.select_font_face(font, slant_dict[slant], weight_dict[weight])

            # position for the text
            Context.move_to(0, _FONT_SIZE)
            # displays the text
            Context.text_path(text)
            Context.set_line_width(1)
            Context.stroke()

        paths = LinearPaths2D.from_svg(_TEMP_FILE_NAME)
        os.remove(_TEMP_FILE_NAME)
        return paths

    @staticmethod
    def raster_image(
        im: npt.NDArray[bool], height: float, stroke_size: float
    ) -> "LinearPaths2D":
        """Rasters an bitmap image into a LinearPaths2D object.

        Rasters an image from top to bottom and switches the direction for each row.
        The first row will be drawn from left ro right.
        While the second row will be drawn from right to left.

        Args:
            img: A two dimensional bitmap. Black, so False will be rasterized.
            height: The desired height of the LinearPaths2D object.
            stroke_size: It determines the density of lines along the y axis.
                A smaller stroke size will result in more lines to raster.

        Returns:
            A rasterized image as a LinearPaths2D object.
        """
        if len(im.shape) != 2:
            raise Exception(
                f"Bitmap is {len(im.shape)} dimensional. Must have two dimensions."
            )

        scaling_factor = height / (im.shape[0] - 1)

        # Generate the list of y values for the raster lines.
        # And the corresponding row index of the actual image.
        y_list = np.linspace(0, height, int(np.ceil(height / stroke_size)))
        row_idx_list = np.round(y_list / scaling_factor).astype(int)

        paths = []
        reverse = False
        for y, row_idx in zip(y_list, row_idx_list):
            row = im[row_idx]
            horizontal_path = []
            segment = []

            # Iterate through the values in the row.
            for col_idx, value in zip(range(len(row)), row):
                if segment:
                    # Segment already is non empty.
                    if value or col_idx == len(row) - 1:
                        # If we find a white pixel
                        # or we reached the end of the image,
                        # we complete the segment.
                        segment.append((col_idx * scaling_factor, y))
                        horizontal_path.append(np.array(segment))
                        segment = []
                else:
                    if not value:
                        # Found a black pixel.
                        # Start at new segment.
                        segment.append((col_idx * scaling_factor, y))

            if reverse:
                # Reverse the horizontal path.
                horizontal_path = [segment[::-1] for segment in horizontal_path[::-1]]
            reverse = not reverse
            paths += horizontal_path

        return LinearPaths2D(paths).vflip().zero()

    @staticmethod
    def make_qrcode(
        data: str, height: float, stroke_size: float, error_correction: str = "L"
    ) -> "LinearPaths2D":
        """Generates a rasterized QR Code path with the given data.

        Args:
            data: The string to be encoded in the QR Code.
            height: The desired height of the LinearPaths2D object.
            stroke_size: It determines the density of lines along the y axis.
                A smaller stroke size will result in more lines to raster.
            error_correction: Controls the error correction used for the QR Code.
                Must be either L, M, Q, or H.
                L corrects  7% or less.
                M corrects 15% or less.
                Q corrects 25% or less.
                H corrects 30% or less.

        Returns:
            A rasterized version of a QR Code as a LinearPaths2D object.
        """
        map_error_str_to_enum = {
            "L": qrcode.constants.ERROR_CORRECT_L,
            "M": qrcode.constants.ERROR_CORRECT_M,
            "Q": qrcode.constants.ERROR_CORRECT_Q,
            "H": qrcode.constants.ERROR_CORRECT_H,
        }

        allowed_error_correction_types = list(map_error_str_to_enum.keys())
        if error_correction not in allowed_error_correction_types:
            raise Exception(
                f"{error_correction} has to be one of the allowed types: {allowed_error_correction_types}"
            )

        # Generate a QR code.
        qr = qrcode.QRCode(
            version=None,
            error_correction=map_error_str_to_enum[error_correction],
            box_size=10,
            border=1,
        )
        qr.add_data(data)
        qr.make(fit=True)

        # Raster image.
        im = np.array(qr.make_image(fill_color="black", back_color="white"))
        return LinearPaths2D.raster_image(im, height, stroke_size)
