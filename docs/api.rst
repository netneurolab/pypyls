.. _api:

.. currentmodule:: pyls

API
===

PLS decompositions
------------------
.. autofunction:: pyls.behavioral_pls
.. autofunction:: pyls.meancentered_pls

PLS results objects
-------------------
.. autoclass:: pyls.structures.PLSResults
.. autoclass:: pyls.structures.PLSPermResults
.. autoclass:: pyls.structures.PLSBootResults
.. autoclass:: pyls.structures.PLSSplitHalfResults
.. autoclass:: pyls.structures.PLSCrossValidationResults
.. autoclass:: pyls.structures.PLSInputs

Results I/O
-----------
.. autofunction:: pyls.save_results
.. autofunction:: pyls.load_results

Matlab compatibility
--------------------
.. autofunction:: pyls.import_matlab_result
