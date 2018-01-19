Installation
============

.. _manual_installation:

Installing the Requirements
----------------------------
| To install PIMPs requirements from command line, please type the following commands on the command line in PIMPs root directory.

.. code-block:: bash

    cat requirements.txt | xargs -n 1 -L 1 pip install

Using argcomplete
-----------------
| PIMP can be used with argcomplete. To enable autocompletion of PIMP
| arguments, add the following line to your .bashrc or .profile:

.. code-block:: bash

    eval "$(register-python-argcomplete pimp)"