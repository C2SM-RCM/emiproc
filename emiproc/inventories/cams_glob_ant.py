"""Compatibility imports for the renamed CAMS global inventory."""

from emiproc.inventories.cams_global import (
    CAMS_FLUX_UNIT,
    CAMS_GLOBAL,
    CAMS_GLOB_ANT,
    CAMSGlobal,
    CAMSGlobAnt,
    CorsoNegativePolicy,
    DEFAULT_CORSO_SECTOR_MAPPING,
    DEFAULT_CORSO_SUBSTANCE_MAPPING,
    SUBSTANCE_ALIASES,
    VOC_COMPONENT_KEYS,
    resolve_cams_substances,
    select_cams_global_variable,
)

__all__ = [
    "CAMS_FLUX_UNIT",
    "CAMS_GLOBAL",
    "CAMS_GLOB_ANT",
    "CAMSGlobal",
    "CAMSGlobAnt",
    "CorsoNegativePolicy",
    "DEFAULT_CORSO_SECTOR_MAPPING",
    "DEFAULT_CORSO_SUBSTANCE_MAPPING",
    "SUBSTANCE_ALIASES",
    "VOC_COMPONENT_KEYS",
    "resolve_cams_substances",
    "select_cams_global_variable",
]
