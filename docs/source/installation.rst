Installation
============


Installing emiproc can be done as a standard python package.

First create a folder for emiproc and go into it 

.. code::

    mkdir emiproc 
    cd emiproc


Clone the repository::

    git clone https://github.com/C2SM-RCM/cosmo-emission-processing.git .


Then install using pip.
Installing as an editable (`-e` options) allows you to directly modify 
the cloned code.

.. code::

    pip install -e .

Many packages will be installed as dependencies.


On Windows 
----------

emiproc works also on windows, but many dependencies are not built for 
windows.

However you can use `pipwin` which will let you install all the dependencies
for windows in the same way `pip` works.

.. code::
    
    pip install pipwin 

    pipwin install geos
    pipwin install geopandas
    pipwin install cartopy 
