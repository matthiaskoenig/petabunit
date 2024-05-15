petabunit - Unit annotations in PEtab
=====================================

|icon1| |icon2| |icon3| |icon4| |icon5| |icon6| |icon7|


.. |icon1| image:: https://github.com/matthiaskoenig/petabunit/workflows/CI-CD/badge.svg
   :target: https://github.com/matthiaskoenig/petabunit/workflows/CI-CD
   :alt: GitHub Actions CI/CD Status
.. |icon2| image:: https://img.shields.io/pypi/v/petabunit.svg
   :target: https://pypi.org/project/petabunit/
   :alt: Current PyPI Version
.. |icon3| image:: https://img.shields.io/pypi/pyversions/petabunit.svg
   :target: https://pypi.org/project/petabunit/
   :alt: Supported Python Versions
.. |icon4| image:: https://img.shields.io/pypi/l/petabunit.svg
   :target: http://opensource.org/licenses/LGPL-3.0
   :alt: GNU Lesser General Public License 3
.. |icon5| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5308801.svg
   :target: https://doi.org/10.5281/zenodo.5308801
   :alt: Zenodo DOI
.. |icon6| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Black
.. |icon7| image:: http://www.mypy-lang.org/static/mypy_badge.svg
   :target: http://mypy-lang.org/
   :alt: mypy

Aim: Provide a means to annotate measurements and observables with units, to facilitate re-use, consistency checks, and automated unit conversion.
Proposal: https://docs.google.com/document/d/1shFaFmykaXUxZqCGNFkRoS5WBPgP5yclWQ7bh9St3N0/edit

License
=======

* Source Code: `LGPLv3 <http://opensource.org/licenses/LGPL-3.0>`__
* Documentation: `CC BY-SA 4.0 <http://creativecommons.org/licenses/by-sa/4.0/>`__

The petabunit source is released under both the GPL and LGPL licenses version 2 or
later. You may choose which license you choose to use the software under.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License or the GNU Lesser General Public
License as published by the Free Software Foundation, either version 2 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

Installation
============
`petabunit` is available from `pypi <https://pypi.python.org/pypi/petabunit>`__ and 
can be installed via:: 

    pip install petabunit

Develop version
---------------
The latest develop version can be installed via::

    pip install git+https://github.com/matthiaskoenig/petabunit.git@develop

Or via cloning the repository and installing via::

    git clone https://github.com/matthiaskoenig/petabunit.git
    cd petabunit
    pip install -e .

To install for development use::

    pip install -e .[development]

© 2024 Matthias König
