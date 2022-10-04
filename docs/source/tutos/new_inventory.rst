Implementing a new Inventory
============================


If you want to include a new inventory in emiproc. 
You should first create a class that inherits from 
:py:class:`~emiproc.inventories.Inventory` .

Make sure your class is as easy as possible to use, such that 
for example you only need to specify the paths of the files.

If you have some parameter to choose (ex. Year of Inventory)
define them in the :py:meth:`__init__` method.