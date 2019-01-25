.. _ref_api:

Reference API
=============

This is the primary reference of ``pyls``. Please refer to the :ref:`user guide
<usage>` for more information on how to best implement these functions in your
own workflows.

.. contents:: **List of modules**
   :local:

.. _ref_decomp:

:mod:`pyls` - PLS decompositions
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

.. _ref_results:

:mod:`pyls.structures` - PLS data structures
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

.. _ref_io:

:mod:`pyls.io` - Data I/O functionality
---------------------------------------

.. automodule:: pyls.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyls.io

.. autosummary::
   :template: function.rst
   :toctree: _generated/

   save_results
   load_results

.. _ref_matlab:

:mod:`pyls.matlab` - Matlab compatibility
-----------------------------------------

.. automodule:: pyls.matlab
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyls.matlab

.. autosummary::
   :template: function.rst
   :toctree: _generated/

   import_matlab_result
