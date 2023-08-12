===================================
Crystal Plasticity Parameterization
===================================

Overview
--------

``matmdl`` is a package implementing a Bayesian `optimization framework`_ using Gaussian process regression to parameterize the hardening models within crystal plasticity finite element method calculations, which are run through a `user material subroutine`_ integrated with Abaqus. 
Its main features are that it:

* Wraps around existing crystal plasticity implementation using Abaqus as the finite element solver
* Uses gradient-free optimization for the multi-dimensional parameter space describing slip system level material properties
* Handles multiple input data sets simultaneously, e.g. mutliple orientations in single crystal deformation
* Considers slight offset orientations in the case of single crystal plasticity, which may go unquantified and unreported in most experimental data sets in the literature

.. _user material subroutine: https://www.researchgate.net/profile/Frank-Richter-5/post/Can_anyone_help_me_te_implement_properly_Huangs_UMAT_for_single_crystal_plasticity_on_Abaqus/attachment/5ca8491b3843b01b9b97ef84/AS%3A744577992499201%401554532635182/download/Huang+-+MECH+178.pdf

.. _optimization framework: https://scikit-optimize.github.io/stable/

.. role:: sh(code)
   :language: bash



Installation
------------
This requires scikit-optimize, which can be installed to a new Conda environment called `opt` with the following steps. 
Configuration files are located in ``src/install/``.

:sh:`conda env create --file=config_simple.yaml`

or by using specific dependency versions, which have been tested:

:sh:`conda env create --file=config_strict.yaml`

.. warning:: Building the documentation requires Sphinx, which remains to be added into the conda environment requirements!


.. Indices
.. -------
.. .. _Document Index: genindex.html
.. .. _Source Index: py-modindex.html

.. * `Document Index`_
.. * `Source Index`_

.. * :ref:`genindex`
.. * :ref:`modindex`


Contents
--------
.. toctree::
   :caption: Contents
   :maxdepth: 2
   :hidden:

   input
   output
   theory
   api
