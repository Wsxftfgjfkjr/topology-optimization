"""HTTP API for the topoopt computational core.

This package is a thin adapter.  Request validation and JSON shaping live here;
every numerical step -- the cantilever problem, the finite element analysis, the
sensitivity filter and the Optimality Criteria loop -- is delegated to
:mod:`topoopt`.  No formula is restated in this layer.
"""
