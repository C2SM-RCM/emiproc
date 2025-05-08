Contribute
==========

We are happy to receive contributions to emiproc.

If you have questions or need help, you are welcome to open an issue on github.

https://github.com/C2SM-RCM/emiproc/issues

We are happy to answer your questions and help you to get started.


Contribute as a developper
--------------------------

If you want to contribute to the code, the prefered way is via pull requests on github.

https://github.com/C2SM-RCM/emiproc/pulls



You can follow the tutorials below to understand how emiproc works and how to
expand it.


Installation and testing
------------------------

To install emiproc for developpement, we recomment that you run 

.. code-block:: bash
    
    pip install -e .[dev]


You can then run the tests with the command, which you should run 
in the main directory of the repository (where the pyproject.toml file is located):

.. code-block:: bash

    pytest


Also, emiproc uses `black <black.readthedocs.io>`_ for code formatting. You can run it with the command:

.. code-block:: bash

    black .




For bulding the documentation, you can run:

.. code-block:: bash

    cd docs
    make html


or on Windows:

.. code-block:: bash

    cd docs
    .\make.bat html



Tutorials for developpers
-------------------------


.. toctree::
    :maxdepth: 2

    new_inventory
    create_grid
