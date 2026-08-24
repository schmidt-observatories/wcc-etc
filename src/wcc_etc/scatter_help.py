import io
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import (
    HistEqStretch,
    LogStretch,
)
from astropy.visualization.mpl_normalize import ImageNormalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RegularGridInterpolator

from .plotting import apply_ax_settings
from .psfsim import FitsImg


class FredResult:
    def __init__(
        self,
        filename=None,
        units="mm",
        data=None,
        desired_power=None,
        peak_scatter_value=None,
        A_AXIS_MIN=0,
        A_AXIS_MAX=1,
        B_AXIS_MIN=0,
        B_AXIS_MAX=1,
    ):
        self.filename = filename
        if filename is not None:
            self.header, self.data = read_fgd(self.filename)
        else:
            self.data = data
            self.header = {}
            self.header["A_AXIS_MIN"] = A_AXIS_MIN
            self.header["A_AXIS_MAX"] = A_AXIS_MAX
            self.header["B_AXIS_MIN"] = B_AXIS_MIN
            self.header["B_AXIS_MAX"] = B_AXIS_MAX
            self.header["A_AXIS_UNITS"] = units
            self.header["B_AXIS_UNITS"] = units
            self.header["DATAUNITS"] = units
            self.filename = ""
        self.basename = os.path.basename(self.filename)

        self.A_AXIS_MIN = float(self.header["A_AXIS_MIN"])
        self.A_AXIS_MAX = float(self.header["A_AXIS_MAX"])
        self.B_AXIS_MIN = float(self.header["B_AXIS_MIN"])
        self.B_AXIS_MAX = float(self.header["B_AXIS_MAX"])
        self.aspect_ratio = (self.A_AXIS_MAX - self.A_AXIS_MIN) / (
            self.B_AXIS_MAX - self.B_AXIS_MIN
        )
        self.A_AXIS_UNITS = self.header["A_AXIS_UNITS"]
        self.B_AXIS_UNITS = self.header["B_AXIS_UNITS"]
        self.DATAUNITS = self.header["DATAUNITS"].replace('"', "")
        self.plate_scale = 4.584  # arcsec/mm
        assert self.A_AXIS_UNITS == self.B_AXIS_UNITS
        assert self.A_AXIS_UNITS == units
        try:
            self.NUMBER_OF_RAYS_USED = int(self.header["NUMBER_OF_RAYS_USED"])
        except (ValueError, KeyError):
            self.NUMBER_OF_RAYS_USED = 0

        self.x = np.linspace(self.A_AXIS_MIN, self.A_AXIS_MAX, self.data.shape[1])
        self.y = np.linspace(self.B_AXIS_MIN, self.B_AXIS_MAX, self.data.shape[0])
        self.delta_x = np.mean(np.diff(self.x))
        self.delta_y = np.mean(np.diff(self.y))
        self.current_power = np.sum(self.data * self.delta_x * self.delta_y)
        self.desired_power = desired_power
        print(f"Current power before any normalization: {self.current_power:0.6f} W")

        if peak_scatter_value is not None:
            self.peak_scatter_value = peak_scatter_value
            print(
                "Not using norm factor, Using peak scatter value:", peak_scatter_value
            )
            self.data = (self.data / np.nanmax(self.data)) * self.peak_scatter_value
        else:
            if desired_power is not None:
                print("Setting to desired power: ", desired_power, "W")
                # self.data = (self.data / np.sum(self.data)) * norm_factor
                self.data = self.data * (desired_power / self.current_power)
                self.current_power = np.sum(self.data * self.delta_x * self.delta_y)
                print("Now integrated flux is: ", self.current_power, "W")
        self.fimg = FitsImg(data=self.data)
        self.x_cut = self.fimg.get_centroid_line_cut(plot=False, line="X")
        self.y_cut = self.fimg.get_centroid_line_cut(plot=False, line="Y")
        print(f"Read file: {self.filename}")

    def plot_data(
        self,
        title="",
        ax=None,
        cmap="viridis",
        stretch="linear",
        origin="lower",
        vmin=None,
        vmax=None,
        colorbar=True,
        plot_sonyimx=True,
        plot_hwk4123=True,
    ):
        if ax is None:
            self.fig, self.ax = plt.subplots(dpi=200, figsize=(12, 4))
        else:
            self.ax = ax
            self.fig = self.ax.get_figure()

        if stretch == "hist":
            print("hist stretch")
            norm = ImageNormalize(stretch=HistEqStretch(self.data))
            self.im = self.ax.imshow(
                self.data,
                cmap=cmap,
                origin=origin,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                extent=[
                    self.A_AXIS_MIN,
                    self.A_AXIS_MAX,
                    self.B_AXIS_MIN,
                    self.B_AXIS_MAX,
                ],
                aspect="auto",
            )
        elif stretch == "log":
            print("log stretch")
            norm = ImageNormalize(self.data, stretch=LogStretch())
            self.im = self.ax.imshow(
                self.data,
                cmap=cmap,
                origin=origin,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                extent=[
                    self.A_AXIS_MIN,
                    self.A_AXIS_MAX,
                    self.B_AXIS_MIN,
                    self.B_AXIS_MAX,
                ],
                aspect="auto",
            )
        else:
            print("linear stretch")
            self.im = self.ax.imshow(
                self.data,
                cmap=cmap,
                origin=origin,
                vmin=vmin,
                vmax=vmax,
                extent=[
                    self.A_AXIS_MIN,
                    self.A_AXIS_MAX,
                    self.B_AXIS_MIN,
                    self.B_AXIS_MAX,
                ],
                aspect="auto",
            )

        # cx, cbar = astropylib.gkastro.ax_add_colorbar(self.ax,p=self.data,cmap=cmap)
        if colorbar:
            cax = inset_axes(
                ax,
                width="5%",
                height="100%",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=ax.transAxes,
                loc="lower left",
                borderpad=0,
            )
            # fig.colorbar(im, cax=cax)
            self.cbar = self.fig.colorbar(self.im, cax=cax)
            self.cbar.ax.set_ylabel(f"{self.DATAUNITS}", fontsize=16)
        self.ax.set_title(title)
        self.ax.set_xlabel(f"X ({self.A_AXIS_UNITS})", fontsize=16)
        self.ax.set_ylabel(f"Y ({self.B_AXIS_UNITS})", fontsize=16)
        _title = (
            f"{self.basename}\n Stretch={stretch}\n"
            f"Integrated Flux={self.current_power:0.6f} W\n"
            f" Set power: {self.desired_power}\n{title}"
        )
        self.ax.set_title(_title, fontsize=16, y=0.95)
        self.ax.minorticks_on()

        # plot a red square that is 36mm x 24mm
        if plot_sonyimx:
            self.ax.add_patch(
                plt.Rectangle(
                    (-18, -12),
                    36,
                    24,
                    edgecolor="red",
                    facecolor="none",
                    lw=2,
                    ls="--",
                    label="IMX 455",
                )
            )

        if plot_hwk4123:
            self.ax.add_patch(
                plt.Rectangle(
                    (-18.8 / 2, -10.58 / 2),
                    18.8,
                    10.58,
                    edgecolor="purple",
                    facecolor="none",
                    lw=2,
                    ls="--",
                    label="HWK4123",
                )
            )

        self.ax.legend(fontsize=12)

    def plot_2panel(self, ylabel=""):
        fig = plt.figure(figsize=(8, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1 / self.aspect_ratio, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        self.plot_data(stretch="hist", ax=ax1, colorbar=True)

        # ax2.plot(self.x,np.log10(self.x_cut),label='X centroid cut',marker='.')
        # ax2.plot(self.y,np.log10(self.y_cut),label='Y centroid cut',marker='.')
        ax2.plot(self.x, self.x_cut, label="X centroid cut", marker=".")
        ax2.plot(self.y, self.y_cut, label="Y centroid cut", marker=".")
        ax2.set_yscale("log")

        ax2.set_xlim(np.min(self.x), np.max(self.x))

        ax2.set_xlabel(f"Distance [{self.A_AXIS_UNITS}]", fontsize=16)
        if ylabel is None:
            ylabel = f"log10(Flux) [{self.DATAUNITS}]"
        else:
            ylabel = "Irradiance [W/mm^2]"
        ax2.set_ylabel(ylabel, fontsize=16)
        apply_ax_settings(ax2)

        XLIM = ax2.get_xlim()

        bx = ax2.twiny()
        ax2.set_xlim(*XLIM)
        bx.set_xlim(*XLIM)
        xticks = [f"{x:0.2f}" for x in ax2.get_xticks() * self.plate_scale]
        bx.set_xticklabels(xticks)
        apply_ax_settings(bx)
        bx.set_xlabel("Distance [arcsec]", fontsize=16)
        # plot Sony IMX extent
        # 36mm x 24mm
        ax2.axvspan(-18, 18, color="red", alpha=0.05, label="IMX 455")
        ax2.axvspan(-18.8 / 2, 18.8 / 2, color="purple", alpha=0.05, label="HWK4123")
        ax2.legend(fontsize=12)

        fig.tight_layout()


def read_fgd(filename, begin_token="BeginData"):
    """
    Read a .fgd file where a variable-length header ends with a line containing
    'BeginData' and the data (2D array) follows.

    Returns:
      header: dict of parsed header key/value pairs (falls back to storing unparsed
              lines under '_lines')
      data:   2D numpy array of the data after 'BeginData'

    EXAMPLE:
        # Example usage (uncomment and adapt path):
        # hdr, arr = read_fgd('../data/example.fgd')
        # print('Header keys:', list(hdr.keys()))
        # print('Data shape:', arr.shape)

    The parser handles header lines with 'key = value' or 'key: value' formats
    and strips common comment characters. Data is loaded with numpy.loadtxt and
    falls back to pandas if necessary.
    """

    header_lines = []
    data_lines = []

    with open(filename) as fh:
        for line in fh:
            # stop when we encounter the BeginData token anywhere on the line
            if begin_token in line:
                break
            header_lines.append(line.rstrip("\n"))
        # collect remaining non-empty lines as data
        for line in fh:
            # skip purely empty lines (but preserve lines with data)
            if line.strip() == "":
                continue
            data_lines.append(line)

    # parse header into a dict
    header = {}
    for ln in header_lines:
        s = ln.strip()
        if not s:
            continue
        # drop common leading comment characters
        s = re.sub(r"^[#;!\/]+", "", s).strip()
        if not s:
            continue
        # split key/value if possible
        if "=" in s:
            k, v = s.split("=", 1)
            header[k.strip()] = v.strip()
        elif ":" in s:
            k, v = s.split(":", 1)
            header[k.strip()] = v.strip()
        else:
            header.setdefault("_lines", []).append(s)

    # load data into numpy array
    if not data_lines:
        data = np.empty((0, 0))
    else:
        txt = "".join(data_lines)
        try:
            data = np.loadtxt(io.StringIO(txt))
        except Exception:
            # fallback to pandas for irregular whitespace parsing
            try:
                import pandas as pd

                data = pd.read_csv(io.StringIO(txt), sep=r"\s+", header=None).values
            except Exception:
                # last resort: return empty array and raw data lines
                data = np.array([])
                header.setdefault("_raw_data", data_lines)

    # ensure 2D array
    if data.size == 0:
        return header, data
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return header, data


def interpolate_matrix(m0, m0_extent, m1, m1_extent, method="cubic"):
    """
    Interpolates a 2D array (m0) onto the grid of another 2D array (m1).

    Parameters:
    -----------
    m0 : numpy.ndarray
        The source 2D array (lower resolution/larger extent).
    m0_extent : tuple
        The physical extent of m0 in the format (x_min, x_max, y_min, y_max).
    m1 : numpy.ndarray
        The target 2D array (only used to get the target shape).
    m1_extent : tuple
        The physical extent of m1 in the format (x_min, x_max, y_min, y_max).
    method : str
        The interpolation method: 'linear' or 'cubic'. Default is 'cubic'.

    Returns:
    --------
    numpy.ndarray
        The interpolated 2D array, matching the exact shape of m1.
    """

    # 1. Extract shapes
    ny0, nx0 = m0.shape
    ny1, nx1 = m1.shape

    x0_min, x0_max, y0_min, y0_max = m0_extent
    x1_min, x1_max, y1_min, y1_max = m1_extent

    # 2. Create the coordinate axes for the source grid (m0)
    # Note: RegularGridInterpolator requires strictly ascending coordinates.
    x_coords0 = np.linspace(x0_min, x0_max, nx0)
    y_coords0 = np.linspace(y0_min, y0_max, ny0)

    # 3. Initialize the interpolator
    # We pass (y, x) because numpy arrays are indexed (row, column)
    # bounds_error=False and fill_value=np.nan ensures that if m1 slightly
    # exceeds m0's bounds due to float precision, it won't crash.
    interpolator = RegularGridInterpolator(
        (y_coords0, x_coords0), m0, method=method, bounds_error=False, fill_value=np.nan
    )

    # 4. Create the coordinate axes for the target grid (m1)
    x_coords1 = np.linspace(x1_min, x1_max, nx1)
    y_coords1 = np.linspace(y1_min, y1_max, ny1)

    # 5. Create a meshgrid of target points
    X1, Y1 = np.meshgrid(x_coords1, y_coords1)

    # Stack them into the shape expected by RegularGridInterpolator: (..., ndim)
    # Again, ordered (Y, X) to match the (row, column) indexing of the data
    target_points = np.stack([Y1, X1], axis=-1)

    # 6. Perform the interpolation
    m1_interpolated = interpolator(target_points)

    return m1_interpolated
