"""Warning and logging suppression utilities."""

from __future__ import annotations

import warnings

from rdkit.rdBase import DisableLog


def suppress_joblib_warnings() -> None:
    """Suppress joblib loky worker timeout warnings.
    
    Call this at module import time to silence joblib warnings about
    workers stopping while jobs are still running.
    """
    warnings.filterwarnings(
        "ignore",
        message="A worker stopped while some jobs were given to the executor",
        category=UserWarning,
        module="joblib.externals.loky.process_executor",
    )


def suppress_rdkit_logs() -> None:
    """Suppress all RDKit logging output.
    
    Call this at module import time to silence RDKit warnings and errors.
    """
    DisableLog("rdApp.*")
