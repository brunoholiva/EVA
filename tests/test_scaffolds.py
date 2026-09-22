"""Tests for generic Murcko scaffold utilities."""

from __future__ import annotations

from chemistry.scaffolds import cluster_by_generic_scaffold, generic_scaffold_hash


class TestGenericScaffoldHash:
    """Tests for generic_scaffold_hash."""

    def test_benzene_returns_generic_ring(self) -> None:
        result = generic_scaffold_hash("c1ccccc1")
        assert result == "C1CCCCC1"

    def test_toluene_shares_scaffold_with_benzene(self) -> None:
        benzene = generic_scaffold_hash("c1ccccc1")
        toluene = generic_scaffold_hash("Cc1ccccc1")
        assert benzene == toluene

    def test_acyclic_returns_acyclic(self) -> None:
        assert generic_scaffold_hash("CCO") == "acyclic"

    def test_invalid_smiles_returns_invalid(self) -> None:
        assert generic_scaffold_hash("not_a_smiles") == "invalid"

    def test_biphenyl_returns_two_ring_scaffold(self) -> None:
        result = generic_scaffold_hash("c1ccc(-c2ccccc2)cc1")
        assert result != generic_scaffold_hash("c1ccccc1")
        assert "C" in result

    def test_hypervalent_sulfur_does_not_crash(self) -> None:
        smi = "COC1CNC(=O)SC(=O)/C(=C(/O)C=O)OC=CS12(=O)NC=C(OC=S)N=C(Oc1ccc(F)cc1-c1cccs1)O2"
        result = generic_scaffold_hash(smi)
        assert isinstance(result, str)
        assert result not in ("invalid", "acyclic", "")


class TestClusterByGenericScaffold:
    """Tests for cluster_by_generic_scaffold."""

    def test_benzene_and_toluene_in_same_cluster(self) -> None:
        clusters = cluster_by_generic_scaffold(["c1ccccc1", "Cc1ccccc1"])
        assert len(clusters) == 1
        assert clusters[0] == {0, 1}

    def test_acyclic_molecules_cluster_together(self) -> None:
        clusters = cluster_by_generic_scaffold(["CCO", "CCCO"])
        assert len(clusters) == 1
        assert clusters[0] == {0, 1}

    def test_different_scaffolds_separate_clusters(self) -> None:
        clusters = cluster_by_generic_scaffold(
            ["c1ccccc1", "CCO", "c1ccc(-c2ccccc2)cc1"]
        )
        assert len(clusters) == 3

    def test_empty_list_returns_empty(self) -> None:
        assert cluster_by_generic_scaffold([]) == []
