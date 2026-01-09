"""*VPRM* : Vegetation Photosynthesis and Respiration Model.


The VPRM model is a parametrized model that estimates
the photosynthesis and respiration of vegetation based on satellite observations.
The model was originally developed by [Mahadevan_2008]_ .

Various extensions of the VPRM model have been implemented in emiproc.

"""

from __future__ import annotations
from enum import Enum
import logging
from typing import Union
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
