"""Plotting helpers for COSI image-deconvolution results."""

import astropy.units as u
import matplotlib.pyplot as plt
from mhealpy import HealpixMap


def plot_reconstructed_image(result, source_position=None):
    """Plot each energy bin in one reconstructed all-sky model.

    Parameters
    ----------
    result : dict
        Result produced by one image-deconvolution iteration.
    source_position : tuple or None
        Expected Galactic longitude and latitude in degrees. When provided,
        the position is marked in red.
    """

    iteration = result["iteration"]
    reconstructed_image = result["model"]

    for energy_index in range(reconstructed_image.axes["Ei"].nbins):
        healpix_map = HealpixMap(
            data=reconstructed_image[:, energy_index],
            unit=reconstructed_image.unit,
        )

        figure, axis = healpix_map.plot("mollview")
        figure.colorbar.set_label(str(reconstructed_image.unit))

        if source_position is not None:
            axis.scatter(
                source_position[0] * u.deg,
                source_position[1] * u.deg,
                transform=axis.get_transform("world"),
                color="red",
            )

        energy_bounds = reconstructed_image.axes["Ei"].bounds[energy_index]
        plt.title(
            f"Iteration = {iteration}, energy index = {energy_index} "
            f"({energy_bounds[0]}–{energy_bounds[1]})"
        )
