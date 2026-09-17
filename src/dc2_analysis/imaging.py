"""Mechanical data and response preparation for COSI imaging analyses."""

from pathlib import Path

import h5py
import numpy as np


VALID_ENERGY_MODES = {"single_bin", "full_dataset"}


def select_imaging_data(data, background, energy_mode, energy_channel=2):
    """Select matching measured-energy bins from data and background.

    Parameters
    ----------
    data, background
        Histograms with a measured-energy axis named ``Em``.
    energy_mode : {"single_bin", "full_dataset"}
        Whether to retain one measured-energy bin or every available bin.
    energy_channel : int
        Zero-based bin index used in ``single_bin`` mode.

    Returns
    -------
    tuple
        Selected data and background histograms.
    """

    _validate_energy_mode(energy_mode)

    if energy_mode == "full_dataset":
        return data, background

    if not 0 <= energy_channel < data.axes["Em"].nbins:
        raise IndexError(
            f"Energy channel {energy_channel} is outside the Em axis."
        )

    energy_slice = slice(energy_channel, energy_channel + 1)
    selected_data = data.slice[{"Em": energy_slice}]
    selected_background = background.slice[{"Em": energy_slice}]

    return selected_data, selected_background


def prepare_imaging_response(
    full_response_path,
    energy_mode,
    energy_channel=2,
    sky_chunk_size=8,
):
    """Return the full response or create a reusable single-bin response.

    The compact response preserves the original histogram and axis metadata.
    Flow-bin padding is removed while the selected regular Ei and Em bins are
    copied. The original response file is always opened read-only.
    """

    _validate_energy_mode(energy_mode)
    full_response_path = Path(full_response_path)

    if energy_mode == "full_dataset":
        return full_response_path

    reduced_response_path = full_response_path.with_name(
        f"{full_response_path.stem}_Ei{energy_channel}_"
        f"Em{energy_channel}{full_response_path.suffix}"
    )

    if reduced_response_path.exists():
        _validate_reduced_response(
            reduced_response_path,
            full_response_path,
            energy_channel,
        )
        return reduced_response_path

    with h5py.File(full_response_path, "r") as source_file:
        source_histogram = source_file["hist"]
        source_contents = source_histogram["contents"]

        if not 0 <= energy_channel < source_contents.shape[1] - 2:
            raise IndexError(
                f"Energy channel {energy_channel} is outside the response."
            )

        # Remove the flow-bin padding from both sky axes and the Phi axis.
        reduced_shape = (
            source_contents.shape[0] - 2,
            1,
            1,
            source_contents.shape[3] - 2,
            source_contents.shape[4] - 2,
        )

        with h5py.File(reduced_response_path, "x") as output_file:
            output_histogram = output_file.create_group("hist")

            for key, value in source_histogram.attrs.items():
                output_histogram.attrs[key] = value

            source_file.copy("hist/axes", output_histogram, name="axes")
            output_axes = output_histogram["axes"]

            for axis_index in (1, 2):
                source_axis = source_file[f"hist/axes/{axis_index}"]
                axis_attributes = dict(source_axis.attrs)
                del output_axes[str(axis_index)]

                reduced_axis = output_axes.create_dataset(
                    str(axis_index),
                    data=source_axis[energy_channel:energy_channel + 2],
                )

                for key, value in axis_attributes.items():
                    reduced_axis.attrs[key] = value

            reduced_contents = output_histogram.create_dataset(
                "contents",
                shape=reduced_shape,
                dtype=source_contents.dtype,
                chunks=(
                    min(sky_chunk_size, reduced_shape[0]),
                    1,
                    1,
                    reduced_shape[3],
                    reduced_shape[4],
                ),
            )

            for sky_start in range(0, reduced_shape[0], sky_chunk_size):
                sky_stop = min(sky_start + sky_chunk_size, reduced_shape[0])
                reduced_contents[sky_start:sky_stop] = source_contents[
                    sky_start + 1:sky_stop + 1,
                    energy_channel + 1:energy_channel + 2,
                    energy_channel + 1:energy_channel + 2,
                    1:-1,
                    1:-1,
                ]

    _validate_reduced_response(
        reduced_response_path,
        full_response_path,
        energy_channel,
    )
    return reduced_response_path


def validate_imaging_energy_axes(data, background, response):
    """Verify that data, background, and response use the same Em edges."""

    data_edges = data.axes["Em"].edges
    background_edges = background.axes["Em"].edges
    response_edges = response.axes["Em"].edges

    background_matches = np.array_equal(data_edges, background_edges)
    response_matches = np.array_equal(data_edges, response_edges)

    if not background_matches or not response_matches:
        raise ValueError(
            "The data, background, and response Em edges must match."
        )

    return {
        "data_background": background_matches,
        "data_response": response_matches,
    }


def _validate_energy_mode(energy_mode):
    if energy_mode not in VALID_ENERGY_MODES:
        choices = ", ".join(sorted(VALID_ENERGY_MODES))
        raise ValueError(
            f"Unknown energy mode {energy_mode!r}. Choose one of: {choices}."
        )


def _validate_reduced_response(
    reduced_response_path,
    full_response_path,
    energy_channel,
):
    """Check the shape and energy edges of an existing compact response."""

    with h5py.File(full_response_path, "r") as source_file:
        source_contents = source_file["hist/contents"]
        expected_shape = (
            source_contents.shape[0] - 2,
            1,
            1,
            source_contents.shape[3] - 2,
            source_contents.shape[4] - 2,
        )
        expected_ei_edges = source_file["hist/axes/1"][
            energy_channel:energy_channel + 2
        ]
        expected_em_edges = source_file["hist/axes/2"][
            energy_channel:energy_channel + 2
        ]

    with h5py.File(reduced_response_path, "r") as reduced_file:
        actual_shape = reduced_file["hist/contents"].shape
        actual_ei_edges = reduced_file["hist/axes/1"][:]
        actual_em_edges = reduced_file["hist/axes/2"][:]

    if actual_shape != expected_shape:
        raise ValueError(
            "The existing reduced response has an unexpected shape: "
            f"{actual_shape}, expected {expected_shape}."
        )

    if not np.array_equal(actual_ei_edges, expected_ei_edges):
        raise ValueError("The reduced response has unexpected Ei edges.")

    if not np.array_equal(actual_em_edges, expected_em_edges):
        raise ValueError("The reduced response has unexpected Em edges.")
