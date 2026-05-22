"""*VPRM* : Vegetation Photosynthesis and Respiration Model.


The VPRM model is a parametrized model that estimates
the photosynthesis and respiration of vegetation based on satellite observations.
The model was originally developed by [Mahadevan_2008]_ .

Various extensions of the VPRM model have been implemented in emiproc.

"""

from __future__ import annotations
from enum import Enum
import logging
from typing import Iterable, Union
import numpy as np
import pandas as pd
import xarray as xr

bandType = Union[np.ndarray, xr.DataArray]


# TODO: once py 3.11 is the minimum version, use StrEnum
class VPRM_Model(Enum):
    """Enum for the VPRM model types.

    - `standard`: Standard VPRM model [Mahadevan_2008]_
    - `urban`: Original Urban VPRM model [Urban_VPRM_Hardiman_2017]_
    - `urban_winbourne`: Urban VPRM model from Winbourne [Urban_VPRM_Winbourne_2021]_
    - `modified_groudji`: Modified VPRM model [VPRM_modified_groudji_2022]_

    """

    standard = "standard"
    urban = "urban"
    urban_winbourne = "urban_winbourne"
    modified_groudji = "modified_groudji"


urban_vprm_models = [
    VPRM_Model.urban,
    VPRM_Model.urban_winbourne,
]


def _series_index_to_numeric(index: pd.Index) -> np.ndarray:
    """Convert series index to a monotonic numeric axis for smoothing."""
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        # Use days to avoid very large numbers from nanoseconds.
        return index.view("i8").astype(float) / 86_400_000_000_000.0
    return np.arange(len(index), dtype=float)


def interpolate_satellite_index_series_lowess(
    series: pd.Series,
    frac: float = 0.1,
    it: int = 3,
    fill_edges: bool = True,
) -> pd.Series:
    """Interpolate a satellite vegetation index timeseries with LOWESS.

    This is a lightweight NumPy implementation of robust LOWESS smoothing,
    inspired by the pyVPRM workflow.

    :param series: Timeseries to process.
    :param frac: Fraction of observations used in each local regression.
        Must be in (0, 1]. Smaller values keep the fit more local and are
        usually better for seasonal satellite series with long gaps.
    :param it: Number of robust reweighting iterations. Must be >= 1.
    :param fill_edges: If True, fill boundary NaNs after smoothing.
    :return: Smoothed/interpolated series with the same index as input.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for LOWESS interpolation. "
            "Please install it with 'pip install statsmodels'."
        ) from e
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1]")
    if it < 1:
        raise ValueError("it must be >= 1")
    if series.empty:
        return series.copy()

    observed = series.dropna()
    if observed.empty:
        return series.copy()
    if observed.size == 1:
        interpolated = series.copy()
        if fill_edges:
            interpolated = interpolated.bfill().ffill()
        return interpolated

    x_all = _series_index_to_numeric(series.index)
    x_obs = _series_index_to_numeric(observed.index)
    y_obs = observed.to_numpy(dtype=float)

    # Standard LOWESS implementation from statsmodels.
    y_all = lowess(
        y_obs,
        x_obs,
        frac=frac,
        it=it,
        is_sorted=True,
        xvals=x_all,
        return_sorted=False,
    )

    out = pd.Series(y_all, index=series.index, dtype=float)
    if fill_edges:
        out = out.bfill().ffill()
    return out


def interpolate_satellite_index_series_kalman(
    series: pd.Series,
    transition_covariance: float = 0.01,
    observation_covariance: float = 0.05,
    initial_state_covariance: float = 1.0,
    fill_edges: bool = True,
) -> pd.Series:
    """Interpolate a satellite vegetation index timeseries with a Kalman smoother.

    Implements a scalar random-walk state model:
    state_t = state_{t-1} + w_t, observation_t = state_t + v_t.

    :param series: Timeseries to process.
    :param transition_covariance: Process noise variance (Q), must be > 0.
    :param observation_covariance: Observation noise variance (R), must be > 0.
    :param initial_state_covariance: Initial covariance, must be > 0.
    :param fill_edges: If True, fill any remaining boundary NaNs.
    :return: Smoothed/interpolated series with the same index as input.
    """
    if transition_covariance <= 0.0:
        raise ValueError("transition_covariance must be > 0")
    if observation_covariance <= 0.0:
        raise ValueError("observation_covariance must be > 0")
    if initial_state_covariance <= 0.0:
        raise ValueError("initial_state_covariance must be > 0")
    if series.empty:
        return series.copy()

    y = series.to_numpy(dtype=float)
    valid = np.isfinite(y)
    if not valid.any():
        return series.copy()

    n = y.size
    x_f = np.full(n, np.nan, dtype=float)
    p_f = np.full(n, np.nan, dtype=float)
    x_pred = np.full(n, np.nan, dtype=float)
    p_pred = np.full(n, np.nan, dtype=float)

    first_valid = int(np.argmax(valid))
    x_prev = y[first_valid]
    p_prev = initial_state_covariance

    for t in range(n):
        x_pr = x_prev
        p_pr = p_prev + transition_covariance

        x_pred[t] = x_pr
        p_pred[t] = p_pr

        if valid[t]:
            k = p_pr / (p_pr + observation_covariance)
            x_upd = x_pr + k * (y[t] - x_pr)
            p_upd = (1.0 - k) * p_pr
        else:
            x_upd = x_pr
            p_upd = p_pr

        x_f[t] = x_upd
        p_f[t] = p_upd
        x_prev = x_upd
        p_prev = p_upd

    # Rauch-Tung-Striebel smoother for the scalar random-walk model.
    x_s = np.copy(x_f)
    p_s = np.copy(p_f)
    for t in range(n - 2, -1, -1):
        denom = p_pred[t + 1]
        if not np.isfinite(denom) or denom <= 0.0:
            continue
        gain = p_f[t] / denom
        x_s[t] = x_f[t] + gain * (x_s[t + 1] - x_pred[t + 1])
        p_s[t] = p_f[t] + gain * gain * (p_s[t + 1] - p_pred[t + 1])

    out = pd.Series(x_s, index=series.index, dtype=float)
    if fill_edges:
        out = out.bfill().ffill()
    return out


def interpolate_satellite_index_series(
    series: pd.Series,
    filter_len: int = 5,
    outlier_threshold: float = 0.2,
    max_filter_duration: pd.Timedelta | str | None = "90D",
    fill_edges: bool = True,
) -> pd.Series:
    """Filter and interpolate a satellite vegetation index timeseries.

    The function matches the original VPRM notebook workflow:

    1. Keep only the observed values of the series.
    2. Apply a rolling-mean based outlier filter on these observations.
    3. Put the filtered observations back on the original sparse timeline.
    4. Interpolate the missing values on the full series.

    :param series: Timeseries to process.
    :param filter_len: Window size used for the rolling mean outlier filter.
        Must be > 0.
    :param outlier_threshold: Relative threshold used to reject outliers.
        Values farther than ``threshold`` from the local rolling mean are set
        to NaN before interpolation.
    :param max_filter_duration: Maximum time distance to an adjacent observation
        before a point is exempted from outlier filtering. This prevents points
        next to long gaps from being filtered against distant observations. Set
        to None to disable this exemption.
    :param fill_edges: If True, apply backward/forward fill after interpolation
        to fill boundary values.
    :return: Interpolated series with the same index as input.
    """
    if filter_len < 1:
        raise ValueError("filter_len must be >= 1")
    if outlier_threshold < 0.0:
        raise ValueError("outlier_threshold must be >= 0")

    if series.empty:
        return series.copy()

    observed = series.dropna()
    if observed.empty:
        return series.copy()
    if observed.size < 2:
        interpolated = series.copy()
        if fill_edges:
            interpolated = interpolated.bfill().ffill()
        return interpolated

    rolling_mean = np.convolve(
        observed.to_numpy(dtype=float),
        np.ones(filter_len, dtype=float) / filter_len,
        mode="same",
    )
    mask_keep = (observed > rolling_mean * (1 - outlier_threshold)) & (
        observed < rolling_mean * (1 + outlier_threshold)
    )
    if max_filter_duration is not None and isinstance(
        observed.index, (pd.DatetimeIndex, pd.TimedeltaIndex)
    ):
        threshold = pd.to_timedelta(max_filter_duration)
        timestamps = observed.index.to_numpy()
        prev_gap = np.abs(timestamps - np.roll(timestamps, 1)).astype("timedelta64[ns]")
        next_gap = np.abs(np.roll(timestamps, -1) - timestamps).astype(
            "timedelta64[ns]"
        )
        prev_gap[0], next_gap[-1] = np.timedelta64("NaT"), np.timedelta64("NaT")
        mask_keep |= (prev_gap > threshold) | (next_gap > threshold)

    filtered_observed = observed.where(mask_keep, np.nan)

    interpolated = series.copy()
    interpolated.loc[observed.index] = filtered_observed
    if filtered_observed.notna().sum() >= 2:
        interpolated = interpolated.interpolate(
            method="akima",
            limit_direction="both",
        )

    if fill_edges:
        interpolated = interpolated.bfill().ffill()

    return interpolated


method_mapping = {
    "akima": interpolate_satellite_index_series,
    "lowess": interpolate_satellite_index_series_lowess,
    "kalman": interpolate_satellite_index_series_kalman,
}


def interpolate_satellite_indices(
    df: pd.DataFrame,
    vegetation_types: Iterable[str] | None = None,
    bands: Iterable[str] = ("evi", "lswi"),
    interpolation_method: str = "akima",
    add_diagnostics: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Interpolate satellite vegetation index columns in a MultiIndex dataframe.

    This helper processes columns following the ``(vegetation_type, band)``
    convention used by VPRM utilities in emiproc.

    :param df: Input dataframe.
    :param vegetation_types: Vegetation types to process. If None, all
        vegetation types that contain requested bands are used.
    :param bands: Index names to interpolate (e.g. ``evi``, ``lswi``).
    :param interpolation_method: Interpolation method key. Available methods:
        ``akima``, ``lowess``, ``kalman``.
    :param add_diagnostics: If True, add ``*_mask`` and ``*_extracted`` columns.
    :param kwargs: Additional keyword arguments passed to
        :py:func:`interpolate_satellite_index_series`.
    :return: Copy of the dataframe with interpolated vegetation indices.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise TypeError("df.columns must be a pandas.MultiIndex")

    out_df = df.copy()
    bands = tuple(bands)

    if vegetation_types is None:
        vegetation_types = []
        for vegetation_type in out_df.columns.get_level_values(0).unique():
            if all((vegetation_type, band) in out_df.columns for band in bands):
                vegetation_types.append(vegetation_type)

    for vegetation_type in vegetation_types:
        for band in bands:
            column = (vegetation_type, band)
            if column not in out_df.columns:
                continue

            extracted = out_df[column].copy(deep=True)

            if add_diagnostics:
                out_df[(vegetation_type, f"{band}_mask")] = extracted.notnull()
                out_df[(vegetation_type, f"{band}_extracted")] = extracted

            if interpolation_method not in method_mapping:
                available = ", ".join(sorted(method_mapping.keys()))
                raise ValueError(
                    f"Unknown interpolation method '{interpolation_method}'. "
                    f"Available methods: {available}."
                )

            out_df[column] = method_mapping[interpolation_method](extracted, **kwargs)

    return out_df


def calculate_vegetation_indices(
    nir: bandType,
    swir: bandType,
    red: bandType,
    blue: bandType,
    # EVI parameters
    vprm_g: float = 2.5,
    vprm_c1: float = 6.0,
    vprm_c2: float = 7.5,
    vprm_l: float = 1.0,
    # Clipping
    clip_evi: bool = False,
):
    """Calculate the vrpm products from the satellite observations.

    The formulas are the following:

    .. math::

        \\mathrm{EVI} &= \\frac{G \\cdot (\\mathrm{NIR} - \\mathrm{RED})}{(\\mathrm{NIR} + C_1 \\cdot \\mathrm{RED} - C_2 \\cdot \\mathrm{BLUE}) + L}

        \\newline

        \\mathrm{LSWI} &= \\frac{\\mathrm{NIR} - \\mathrm{SWIR}}{\\mathrm{NIR} + \\mathrm{SWIR}}

        \\newline

        \\mathrm{NDVI} &= \\frac{\\mathrm{NIR} - \\mathrm{RED}}{\\mathrm{NIR} + \\mathrm{RED}}



    The input bands can by numpy arrays or xarray DataArrays.

    :param nir: Near Infrared band
    :param swir: Shortwave Infrared band
    :param red: Red band
    :param blue: Blue band

    :param vprm_g: Gain factor for EVI
    :param vprm_c1: Coefficient 1 for EVI
    :param vprm_c2: Coefficient 2 for EVI
    :param vprm_l: Coefficient L for EVI

    :param clip_evi: Clip the EVI values between 0 and 1.
        As the equation for EVI does not produce a proper index,
        values can be negative or above 1 if not clipped.
    :return: Tuple with the EVI, LSWI and NDVI

    """
    evi = vprm_g * (nir - red) / (nir + vprm_c1 * red - vprm_c2 * blue + vprm_l)
    lswi = (nir - swir) / (nir + swir)
    ndvi = (nir - red) / (nir + red)

    if clip_evi:
        evi = np.clip(evi, 0, 1)

    return evi, lswi, ndvi


def calculate_vprm_emissions(
    df: pd.DataFrame,
    df_vprm: pd.DataFrame,
    model: VPRM_Model | str = VPRM_Model.standard,
) -> pd.DataFrame:
    """Calculate the emissions using the VPRM model.

    This function uses timeseries of vegetation indices, temperature and radiation
    to calculate the respiration and photosynthesis emissions of vegetation.

    For more details about the VPRM model, see :ref:`vprm` .


    :param df: Dataframe with the observations. It must be a multiindex dataframe with the following columns:

        - `RAD`: Shortwave radiation in W/m2
        - ('T', 'global'): Temperature in degC
        - (vegetation_type, 'lswi'): Land Surface Water Index
        - (vegetation_type, 'evi'): Enhanced Vegetation Index

        Urban VPRM models:

        - (vegetation_type, 'evi_ref'): Reference EVI for the urban VPRM model.
            This is the EVI at a non-urban reference site
            representing a baseline leaf-off, woody biomass respiration.
        - ('T', 'urban'): Temperature in degC in the urban area,
            representing the urban heat island effect.


    :param df_vprm: Dataframe with the VPRM parameters.
        Each row must correspond to a vegetation type and have the following columns:

        - `alpha`: Respiration parameter
        - `beta`: Respiration parameter
        - `lambda`: Photosynthesis parameter
        - `Tmin`: Minimum temperature for photosynthesis
        - `Topt`: Optimal temperature for photosynthesis
        - `Tmax`: Maximum temperature for photosynthesis
        - `Tlow`: Low temperature for photosynthesis
        - `PAR0`: Photosynthetically Active Radiation parameter

        Urban VPRM models:

        - `isa`: Impervious Surface Area (ISA) at the vegetation location.
            This is the fraction of the area that is impervious (e.g. buildings, roads, etc.)
            Use 0.5 if you don't know.

        Modified VPRM:

        - `theta1`: Coeff for water respiration scaling factor
        - `theta2`: Coeff for water respiration scaling factor
        - `theta3`: Coeff for water respiration scaling factor
        - `alpha1`: Respiration parameter
        - `alpha2`: Respiration parameter
        - `gamma`: Coeff for EVI in respiration
        - `Tcrit`: critical temperature for respiration
        - `Tmult`: value between 0-1 to weigh the difference between atm temp and tcrit

    :param model: VPRM model to use. See :py:class:`VPRM_Model` for the list of models.

    :return: Dataframe with the emissions. Some columns are added

        - (vegetation_type, 'resp_min'): Respiration at the minimum temperature
        - (vegetation_type, 'resp_max'): Respiration at the maximum temperature
        - (vegetation_type, 'resp'): Respiration
        - (vegetation_type, 'gee'): Gross Ecosystem Exchange
        - (vegetation_type, 'nee'): Net Ecosystem Exchange (nee = gee - resp)
        - (vegetation_type, 'Tscale'): Temperature scale
        - (vegetation_type, 'Wscale'): Water scale
        - (vegetation_type, 'Pscale'): Photosynthesis scale


        Urban VPRM models:

        - (vegetation_type, 'resp_h'): Heterotrophic respiration
        - (vegetation_type, 'resp_a'): Autotrophic respiration
    """
    logger = logging.getLogger(__name__)
    df = df.copy()

    model = VPRM_Model(model)

    df_vprm["resp_min"] = df_vprm["alpha"] * df_vprm["Tlow"] + df_vprm["beta"]

    # Photosynthetically Active Radiation (PAR, μmol m−2 s−1)
    # Conversion from original vprm paper, assuming RAD is shortwave radiation
    df["PAR"] = df["RAD"] / 0.505

    if model in urban_vprm_models:
        # Ensure that the urban temperature is present
        if ("T", "urban") not in df.columns:
            raise KeyError(
                "Urban VPRM is activated but the urban temperature is missing in the dataframe. "
                "Please add the ('T', 'urban') column to the dataframe."
            )

    if (df_vprm["lambda"] < 0.0).any():
        logger.warning(
            "Some lambda values in the VPRM parameters are negative. "
            "Emiproc expects lambda to be positive. "
            "If you encounter positive GEE values, check your lambda values."
        )

    for vegetation_type in df_vprm.index:
        if not all(
            [(vegetation_type, index) in df.columns for index in ["lswi", "evi"]]
        ):
            logger.warning(
                f"Missing {vegetation_type} in the observation dataframe, skipping"
            )
            continue

        # Add to the metot the paramters from the satellite observations
        # Use interpolation to get the values for the missing dates
        lswi = df[(vegetation_type, "lswi")]
        evi = df[(vegetation_type, "evi")]

        Tmin = df_vprm.loc[vegetation_type, "Tmin"]
        Topt = df_vprm.loc[vegetation_type, "Topt"]
        Tmax = df_vprm.loc[vegetation_type, "Tmax"]
        Tlow = df_vprm.loc[vegetation_type, "Tlow"]

        if model == VPRM_Model.urban_winbourne:
            # For T scale, the vegetation specific T parameters are not used
            Tmin, Tmax = 0.0, 40.0

        alpha = df_vprm.loc[vegetation_type, "alpha"]
        beta = df_vprm.loc[vegetation_type, "beta"]

        # Get correct temperature serie
        temperature = df[("T", "urban" if model in urban_vprm_models else "global")]

        # Calculate the respiration
        resp = alpha * temperature + beta

        # for respiration use the modified VPRM if requested
        if model == VPRM_Model.modified_groudji:
            alpha1 = df_vprm.loc[vegetation_type, "alpha1"]
            alpha2 = df_vprm.loc[vegetation_type, "alpha2"]
            gamma = df_vprm.loc[vegetation_type, "gamma"]
            k1 = df_vprm.loc[vegetation_type, "theta1"]
            k2 = df_vprm.loc[vegetation_type, "theta2"]
            k3 = df_vprm.loc[vegetation_type, "theta3"]
            Tcrit = df_vprm.loc[vegetation_type, "Tcrit"]
            Tmult = df_vprm.loc[vegetation_type, "Tmult"]

            wscale2 = (lswi - np.nanmin(lswi)) / (np.nanmax(lswi) - np.nanmin(lswi))

            # modified air temperature variable intended to capture soil temperatures
            # that remain warmer than air temperatures in winter
            temp_mod = temperature.where(
                temperature >= Tcrit,
                other=Tcrit - Tmult * (Tcrit - temperature),
            )

            resp = (
                beta
                + alpha1 * temp_mod
                + alpha2 * temp_mod**2
                + gamma * evi
                + k1 * wscale2
                + k2 * wscale2 * temp_mod
                + k3 * wscale2 * temp_mod**2
            )

        # Under Tlow, use a constant value
        mask_low_T = temperature <= Tlow

        # Set T = Tlow when T < Tlow to account for the persistence
        # of soil respiration in winter, when air temperatures are very cold
        # but soils remain warm
        resp_min = alpha * Tlow + beta

        resp.loc[mask_low_T] = resp_min

        if model in urban_vprm_models:
            # Split the urban vegetation into two parts
            # initial ecosystem respiration (autotrophic + heterotrophic)
            resp_e_init = alpha * temperature + beta
            df[(vegetation_type, "resp_e_init")] = resp_e_init
            # Heterotrophic respiration
            # isa = impervious surface areas
            isa = df_vprm.loc[vegetation_type, "isa"]
            resp_h = (1 - isa) * resp_e_init / 2.0

            # Get reference  the yearly minimum of EVI at a reference Forest
            # min of evi_ref is representing leaf-off,
            # woody biomass autotrophic respiration
            evi_ref = df[(vegetation_type, "evi_ref")]
            resp_a = (evi + np.nanmin(evi_ref) * isa) / evi_ref * resp_e_init / 2.0

            df[(vegetation_type, "resp_h")] = resp_h
            df[(vegetation_type, "resp_a")] = resp_a

            # Bring the two components together
            resp = resp_h + resp_a

        # GEE
        # GEE is calculated from various sub components

        # Temperature scale
        Tprod = (temperature - Tmin) * (temperature - Tmax)
        Tscale = Tprod / (Tprod - (temperature - Topt) ** 2)
        Tscale[temperature <= Tmin] = 0.0
        if model == VPRM_Model.urban_winbourne:
            mask_low_T = temperature <= 20
            Tscale.loc[mask_low_T] = Tprod / (Tprod - (temperature - 20) ** 2)
            mask_mid_T = (temperature >= 20) & (temperature <= 30)
            Tscale.loc[mask_mid_T] = 1.0
            mask_high_T = temperature >= 30
            Tscale.loc[mask_high_T] = Tprod / (Tprod - (temperature - 30) ** 2)

        df[(vegetation_type, "Tscale")] = Tscale

        # Water scale
        Wscale = (1 + lswi) / (1 + np.nanmax(lswi))
        df[(vegetation_type, "Wscale")] = Wscale

        # Photosynthesis scale

        # to detect phase two occurrence let's use a EVI threshold method
        # see WRF-GHG
        # https://github.com/wrf-model/WRF/blob/f34b11dbb89c002c5c0dca1195aab35daeed7349/chem/module_ghg_fluxes.F#L199
        # see pyVPRM
        # https://github.com/tglauch/pyVPRM/blob/308421b3f1ade445fef1b9edc37547db83a295cb/pyVPRM/VPRM.py#L561
        # since it's not simple to get vegetation dynamics on Sentinel2
        # (while it's available for MOD12Q2 used by Mahadevan et al., 2008)
        # the overall max and min of EVI is used
        # not the EVI max/min during growing phase only (as it should be).
        evithr = np.nanmin(evi) + 0.55 * (np.nanmax(evi) - np.nanmin(evi))

        if model in urban_vprm_models:
            # Simpler EVI formulation in urban VPRM
            Pscale = (evi - np.nanmin(evi)) / (np.nanmax(evi) - np.nanmin(evi))
        else:
            # bud-burst to full canopy period
            Pscale = (1 + lswi) / 2.0
            # is 1 during phase two (Mahadevan et al, paragraph [14])
            Pscale[evi >= evithr] = 1.0

        # for evergreen, Pscale is 1 fixed (Mahadevan et al, paragraph [13])
        veg_type_str = str(vegetation_type).lower()
        if "evergreen" in veg_type_str:
            Pscale = 1.0

        df[(vegetation_type, "Pscale")] = Pscale

        gee = -(
            df_vprm.loc[vegetation_type, "lambda"]
            * Tscale
            * Pscale
            * Wscale
            * evi
            * df["PAR"]
            / (1 + df["PAR"] / df_vprm.loc[vegetation_type, "PAR0"])
        )

        #  VPRM produces umoles/m2/s
        df[(vegetation_type, "resp")] = resp
        df[(vegetation_type, "gee")] = gee
        df[(vegetation_type, "nee")] = resp + gee

    return df
