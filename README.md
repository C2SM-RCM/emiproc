# Emiproc
**Python package for processing emission datasets and preparing them for atmospheric transport models.**

[![DOI](https://joss.theoj.org/papers/10.21105/joss.07509/status.svg)](https://doi.org/10.21105/joss.07509)
[![ReadTheDocs](https://readthedocs.org/projects/emiproc/badge/?version=master)](emiproc.readthedocs.io)


![Emission Processing Pipeline](docs/source/diagrams/pipeline.drawio.svg)

`emiproc` helps scientists convert, process, and prepare gridded emissions datasets
for atmospheric modeling, data visualization, and scientific deliverables.

If you need to prepare emissions for a model and you don't want to get 
a headache with regridding, sector conversion or temporal distributions,
`emiproc` is for you.


## 📦 Installation 

```bash
pip install emiproc
```

## 📚 Documentation

For a quick start we recommend [the EDGAR tutorial](https://emiproc.readthedocs.io/en/master/tutos/edgar_processing.html#EDGAR-Inventory-Processing)

Full documentation: https://emiproc.readthedocs.io .


### 💨 Supported Models 

* [ICON-ART](https://www.icon-art.kit.edu/)
* [WRF-Chem](https://www2.acom.ucar.edu/wrf-chem)
* [GRAMM-GRAL](https://gral.tugraz.at/)
* [NetCDF](https://emiproc.readthedocs.io/en/master/api/exports.html#emiproc.exports.rasters.export_raster_netcdf)

### 🌍 Suported Inventories 

* [EDGAR](https://edgar.jrc.ec.europa.eu/) – Global anthropogenic emissions
* [TNO](https://airqualitymodeling.tno.nl/emissions/) – European emissions
* [GFAS](https://atmosphere.copernicus.eu/global-fire-emissions) – Fire emissions
* [GFED](https://www.globalfiredata.org/) – Global fire emissions

### 🏭 Emission Models 

* [VPRM](https://doi.org/10.1029/2006GB002735) - Vegetation
* [HDD](https://en.wikipedia.org/wiki/Heating_degree_day) - Domestic Heating 
* [Human Respiration](https://emiproc.readthedocs.io/en/master/api/models.html#module-emiproc.human_respiration)

## 🙋 Need help or want to contribute?

If you’d like to support a new model, emission inventory, or temporal profile, feel free to 
[open an issue](https://github.com/C2SM-RCM/emiproc/issues) to start a discussion.
We're happy to help and collaborate!


## 🪪 License

This work is licensed under a BSD-3-Clause licence. See the LICENSE file for details or https://opensource.org/license/bsd-3-clause


## 📑 References 

If you use `emiproc` in your research or project, 
please cite the following publication: 

> Lionel et al., (2025).  
> Emiproc: A Python package for emission inventory processing.  
> Journal of Open Source Software, 10(105), 7509  
> [https://doi.org/10.21105/joss.07509](https://doi.org/10.21105/joss.07509)



