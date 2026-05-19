# -*- coding: utf-8 -*-
"""Entrypoint for the main eGeMAPS baseline/deviation CBM experiment.

The implementation lives in the ``ser_cbm`` package so the code release is easier
to inspect and reuse. This wrapper is kept for backward-compatible commands:

    python main_egemaps_baseline_deviation_cbm.py
"""

from ser_cbm import *  # noqa: F401,F403


if __name__ == "__main__":
    run_experiment(CFG)
