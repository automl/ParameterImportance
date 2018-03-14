Installation
============

.. _manual_installation:

Installing the Requirements
----------------------------
| To install PyImps requirements from command line, please type the following commands on the command line in PyImps root directory.

.. code-block:: bash

    cat requirements.txt | xargs -n 1 -L 1 pip install
    python setup.py install

After the installation you can call PyImp from anywhere with the keyword pyimp.

Using argcomplete
-----------------
| PyImp can be used with argcomplete. To enable autocompletion of PyImp
| arguments, add the following line to your .bashrc or .profile:

.. code-block:: bash

    eval "$(register-python-argcomplete pyimp)"