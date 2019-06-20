# Online emission processing

Preprocessing of the emissions for the online emission module of cosmo.
Produces gridded annual emissions as well as temporal and vertical profiles.

## Notice

This repository is a merge of

https://gitlab.empa.ch/abt503/users/hjm/online-emission-processing

https://gitlab.empa.ch/abt503/users/jae/online-emission-processing

https://gitlab.empa.ch/abt503/users/muq/online_emission_cosmoart

and will be hosted at

https://github.com/C2SM-RCM/cosmo-emission-processing.

It deliberately is not a fork of any of the existing repositories since that would introduce an
unwanted dependency. However, the commit history will be kept intact as much as possible.

## Formatting

This code is formatted using [black](https://black.readthedocs.io/en/stable/).
Run `find . -name '*.py' -exec black -l 80 --target-version py36 {} +` before commiting
to format your changes before commiting.

## License

This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
