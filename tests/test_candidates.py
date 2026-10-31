"""Tests for CandidateStore hit and scaffold tracking."""

from __future__ import annotations

from optimization.candidates import CANDIDATE_COLUMNS, CandidateStore


def test_candidate_store_ranks_and_clusters_scaffolds() -> None:
    store = CandidateStore(hit_threshold=0.5)

    # Below threshold -> not stored in records, no >=0.5 hit
    store.record_scored(
        canonical_smiles="CCO",
        smiles="OCC",
        p_active=0.30,
        walker_id=0,
        step=0,
        oracle_call=1,
    )
    # Benzene derivative (generic scaffold C1CCCCC1) at 0.65
    store.record_scored(
        canonical_smiles="Cc1ccccc1",
        smiles="Cc1ccccc1",
        p_active=0.65,
        walker_id=1,
        step=0,
        oracle_call=2,
    )
    # Second benzene derivative (same generic scaffold) at 0.85
    store.record_scored(
        canonical_smiles="CCc1ccccc1",
        smiles="CCc1ccccc1",
        p_active=0.85,
        walker_id=2,
        step=1,
        oracle_call=3,
    )
    # Biphenyl (different generic scaffold) at 0.70
    store.record_scored(
        canonical_smiles="c1ccc(-c2ccccc2)cc1",
        smiles="c1ccc(-c2ccccc2)cc1",
        p_active=0.70,
        walker_id=0,
        step=1,
        oracle_call=4,
    )
    # Hit between 0.5 and 0.6 -> stored in records, but not counted in >=0.6 scaffolds
    store.record_scored(
        canonical_smiles="CCN",
        smiles="CCN",
        p_active=0.55,
        walker_id=3,
        step=2,
        oracle_call=5,
    )

    assert len(store) == 4
    assert store.best_p_active == 0.85
    assert store.n_hits_05 == 4
    assert store.n_hits_06 == 3
    assert store.n_hits_08 == 1
    assert store.unique_scaffolds_06 == 2
    assert abs(store.largest_scaffold_frac_06 - (2.0 / 3.0)) < 1e-9

    df = store.to_dataframe()
    assert list(df.columns) == CANDIDATE_COLUMNS
    assert df["p_active"].tolist() == [0.85, 0.70, 0.65, 0.55]
    assert df["rank"].tolist() == [1, 2, 3, 4]


def test_empty_candidate_store_dataframe() -> None:
    store = CandidateStore(hit_threshold=0.6)
    df = store.to_dataframe()
    assert df.empty
    assert list(df.columns) == CANDIDATE_COLUMNS
    assert store.largest_scaffold_frac_06 == 0.0
