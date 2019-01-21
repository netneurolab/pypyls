.. _api:

Reference API
=============

This is the primary reference of ``pyls``. Please refer to the :ref:`user guide
<usage>` for more information on how to best implement these functions in your
own workflows.

.. _decomp_ref:

:mod:`pyls` - PLS Decompositions
--------------------------------------

.. automodule:: pyls.types
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyls

.. autosummary::
   :template: function.rst
   :toctree:  _generated/

    behavioral_pls
    meancentered_pls

.. _results_ref:

:mod:`pyls.structures` - PLS Results Objects
--------------------------------------------

.. automodule:: pyls.structures
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyls.structures

.. autosummary::
   :template: class.rst
   :toctree: _generated/

   PLSResults
   PLSPermResults
   PLSBootResults
   PLSSplitHalfResults
   PLSCrossValidationResults
   PLSInputs

.. _io_ref:

:mod:`pyls.io` - Data I/O
-------------------------

.. automodule:: pyls.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyls.io

.. autosummary::
   :template: function.rst
   :toctree: _generated/

   save_results
   load_results

.. _matlab_ref:

:mod:`pyls.matlab` - Matlab Compatibility
-----------------------------------------

.. automodule:: pyls.matlab
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyls.matlab

.. autosummary::
   :template: function.rst
   :toctree: _generated/

   import_matlab_result
