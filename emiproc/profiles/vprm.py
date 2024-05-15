"""*VPRM* : Vegetation Photosynthesis and Respiration Model.


The VPRM model is a parametrized model that estimates
the photosynthesis and respiration of vegetation based on satellite observations.
The model was developed by [Mahadevan_2008]_

"""
from __future__ import annotations
import logging
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr

bandType = Union[np.ndarray,xr.DataArray]


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



def calculate_vprm_emissions(df: pd.DataFrame, df_vprm: pd.DataFrame) -> pd.DataFrame:
    """Calculate the emissions using the VPRM model.

    This function uses timeseries of vegetation indices, temperature and radiation 
    to calculate the
    respiration and photosynthesis emissions of vegetation.

    It handles various vegetation types.
    Each vegetation type has its own parameters for the VPRM model.

    It also includes extensions of the VPRM model for urban areas.


    The equations used are the following:

    PAR (Photosynthetically Active Radiation) is calculated from the shortwave radiation:
    
    .. math::
        \\mathrm{PAR} = \\frac{\\mathrm{RAD}}{0.505}
    
    Respiration is calculated from the temperature:
    
    .. math::
        \\mathrm{Resp} = \\alpha * T + \\beta
    
    The Gross Ecosystem Exchange (GEE) is calculated from the temperature, PAR and the vegetation indices:
    
    .. math::
        \\mathrm{GEE} = \\lambda * T_{scale} * P_{scale} * W_{scale} * \\mathrm{EVI} * \\frac{ \\mathrm{PAR} }{1 + \\frac{\\mathrm{PAR}}{PAR0}}

    where the different scales are:

    - :math:`T_{scale}`: Temperature scale

    .. math::
        T_{\\text{scale}} = \\frac{(T - T_{\\text{min}}) \\cdot (T - T_{\\text{max}})}{(T - T_{\\text{min}}) \\cdot (T - T_{\\text{max}}) + (T - T_{\\text{opt}})^2} \\text{if } T \\geq T_{\\text{min}} \\text{ else } 0
        


    - :math:`P_{scale}`: Photosynthesis scale

    .. math::
        P_{scale} = \\frac{1 + \\mathrm{LSWI}}{2}
    
    - :math:`W_{scale}`: Water scale

    .. math::
        W_{scale} = \\frac{1 + \\mathrm{LSWI}}{1 + \\mathrm{LSWI}_{max}}

    The Net Ecosystem Exchange (NEE) is calculated from the respiration and GEE.
    
    .. math::
        \\mathrm{NEE} = \\mathrm{Resp} + \\mathrm{GEE}

        
    Units for all fluxes (NEE, GEE, Resp, ...) are

    .. math:: 
        \\frac{\\mu mol_{\\mathrm{CO2}}}{m^2 * s}

        
    Urban modifications

    The VPRM model can be extended to urban areas according to  [Urban_VPRM]_ .
    
    - A "urban temperature" is used instead of the global temperature to represent
        the urban heat island phenomenon.
    - The formula for :math:`P_{scale}` is modified to

    .. math::
        P_{scale} = \\frac{\\mathrm{EVI} - \\mathrm{EVI}_{min}}{\\mathrm{EVI}_{max} - \\mathrm{EVI}_{min}}

    - The respiration is calculated differently
    
    .. math::
        \\mathrm{Resp} = \\frac{\\mathrm{Resp_{e-init}}}{2} * (1 - \\mathrm{ISA}) + \\frac{\\mathrm{EVI} + \\mathrm{EVI}_{min} * \\mathrm{ISA}}{\\mathrm{EVI}_{ref}} * \\frac{\\mathrm{Resp_{e-init}}}{2}
    
    where :math:`\\mathrm{Resp_{e-init}}` is the basic vprm respiration and :math:`\\mathrm{ISA}` is the impervious surface area at the vegetation location.
    
    .. warning::
        The urban VPRM model is currently not fully implemented.
    
    
    :param df: Dataframe with the observations. It must be a multiindex dataframe with the following columns:
    
        - `RAD`: Shortwave radiation in W/m2
        - ('T', 'global'): Temperature in degC
        - (vegetation_type, 'lswi'): Land Surface Water Index 
        - (vegetation_type, 'evi'): Enhanced Vegetation Index
        - ('T', 'urban'): Optional for urban VPRM. Temperature in degC (urban area)
    
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
        - `is_urban`: Boolean indicating if the vegetation type is urban (optional, default is False)
   
    :return: Dataframe with the emissions. Some columns are added

        - (vegetation_type, 'resp_min'): Respiration at the minimum temperature
        - (vegetation_type, 'resp_max'): Respiration at the maximum temperature
        - (vegetation_type, 'resp'): Respiration
        - (vegetation_type, 'gee'): Gross Ecosystem Exchange
        - (vegetation_type, 'nee'): Net Ecosystem Exchange (nee = gee - resp)
    """
    logger = logging.getLogger(__name__)
    df = df.copy()

    df_vprm['resp_min'] = df_vprm['alpha'] * df_vprm["Tlow"] + df_vprm['beta']

    # Photosynthetically Active Radiation (PAR, μmol m−2 s−1) 
    # Conversion from orginal vprm paper, assuming RAD is shortwave radiation
    df['PAR'] = df['RAD'] / 0.505


    if 'is_urban' not in df_vprm.columns:
        df_vprm['is_urban'] = False
    if any(df_vprm['is_urban']):
        # Ensure that the urban temperature is present
        if ('T', 'urban') not in df.columns:
            raise ValueError("Urban VPRM is activated but the urban temperature is missing in the dataframe")

    for vegetation_type in df_vprm.index:
        if not all([(vegetation_type, index) in df.columns for index in ['lswi', 'evi']]):
            logger.warning(f"Missing {vegetation_type} in the observation dataframe, skipping")
            continue


        # Add to the metot the paramters from the satellite observations 
        # Use interpolation to get the values for the missing dates
        lswi = df[(vegetation_type, 'lswi')]
        evi = df[(vegetation_type, 'evi')]

        
            
        is_urban = df_vprm.loc[vegetation_type, 'is_urban']

        Tmin = df_vprm.loc[vegetation_type, 'Tmin']
        Topt = df_vprm.loc[vegetation_type, 'Topt']
        Tmax = df_vprm.loc[vegetation_type, 'Tmax']
        Tlow = df_vprm.loc[vegetation_type, 'Tlow']

        # Get correct temperature df
        if is_urban:
            temperature = df[('T', 'urban')]
        else:
            temperature = df[('T', 'global')]
        
        # Calculate the respiration 

        # Resp = alpha * T + beta
        resp  = df_vprm.loc[vegetation_type, 'alpha'] * temperature + df_vprm.loc[vegetation_type, 'beta']

        # Under t low, use a contsant value 
        mask_low_T = temperature <= Tlow
        resp.loc[mask_low_T] =  df_vprm.loc[vegetation_type, 'resp_min']



        ## Split the urban vegetation into two parts
        ## initial ecosystem respiration (authotropphic + heterotropohic)
        #df[(vegetation_type, 'resp_e_init')] = df[(vegetation_type, 'resp _urban')] 
    #
        ## Heterotrophic respiration
        ## isa = impervious surface areas 
        #isa = 0.5 # ??? not  sure what value this should be
        #r  = (1- isa) * df[(vegetation_type, 'resp_e_init')] / 2.
    #
        ## Get reference  the yearly minimum of EVI at a reference Forest 
        ## (representing leaf-off, woody biomass autotrophic respiration
        #evi_ref = df_means[(vegetation_type_ref[vegetation_type], 'EVI')]
        #r_a = (evi + np.nanmin(evi_ref) * isa) / evi_ref * resp_e_init / 2.
        #
        ## Bring the two components together
        #resp  = r  + r_a



        # GEE
        Tscale  = (temperature - Tmin) * (temperature- Tmax)
        Tscale  = Tscale  / (Tscale  - (temperature - Topt) ** 2)
        Tscale [temperature <= Tmin] = 0.0
        df[(vegetation_type, 'Tscale')] = Tscale 


        # Typical summer values for LSWI, EVI and Wscale
        # To be replaced with satellite based parameters
        Wscale  = (1 + lswi) / (1 + np.nanmax(lswi))
        df[(vegetation_type, 'Wscale')] = Wscale

        if is_urban:
            # Simpler EVI forumalation in urban VPRM
            Pscale  = (evi - np.nanmin(evi)) / (np.nanmax(evi) - np.nanmin(evi))
        else:
            Pscale  = (1 + lswi) / 2.0
        df[(vegetation_type, 'Pscale')] = Pscale

        gee  = (
            df_vprm.loc[vegetation_type, "lambda"]
            * Tscale 
            * Pscale 
            * Wscale 
            * evi
            * df['PAR']
            / (1 + df['PAR'] / df_vprm.loc[vegetation_type, "PAR0"])
        )


        #  VPRM produces umoles/m2/s
        df[(vegetation_type, 'resp')] = resp   
        df[(vegetation_type, 'gee')] = gee   
        df[(vegetation_type, 'nee')] = resp  + gee   
    
    return df
