"""The API map must stay exactly in sync with `__all__`.

A grouped index of the public API is only useful if it is *complete and current*. A note
asking future contributors to update it would rot; this makes the sync mechanical, so a
new export cannot quietly go unlisted and a removed one cannot linger.

That distinction is the point: the value of the map is that you can trust it covers
everything. A map that is 95% right is worse than no map, because a reader stops looking
after it.
"""

from __future__ import annotations

import dyf
from dyf._api_map import API_GROUPS, NOT_REEXPORTED, overview


def _mapped_names() -> list[str]:
    return [name for group in API_GROUPS.values() for name in group.names]


class TestCoverage:
    def test_every_export_is_grouped(self):
        missing = sorted(set(dyf.__all__) - set(_mapped_names()))
        assert not missing, f"exports missing from the API map: {missing}"

    def test_no_phantom_names(self):
        extra = sorted(set(_mapped_names()) - set(dyf.__all__))
        assert not extra, f"API map lists names that are not exported: {extra}"

    def test_no_name_listed_twice(self):
        names = _mapped_names()
        dupes = sorted({n for n in names if names.count(n) > 1})
        assert not dupes, f"names appearing in more than one group: {dupes}"

    def test_every_mapped_name_actually_resolves(self):
        """Catches a name that is exported and grouped but does not import."""
        broken = [n for n in _mapped_names() if not hasattr(dyf, n)]
        assert not broken, f"grouped names that do not resolve on the package: {broken}"


class TestGroupQuality:
    def test_every_group_has_a_summary(self):
        empty = [name for name, g in API_GROUPS.items() if not g.summary.strip()]
        assert not empty

    def test_start_here_is_a_member_of_its_own_group(self):
        """An entry point pointing outside its group would send a reader the wrong way."""
        for name, g in API_GROUPS.items():
            if g.start_here is not None:
                assert g.start_here in g.names, f"{name}.start_here is not in the group"

    def test_no_group_is_empty(self):
        assert all(g.names for g in API_GROUPS.values())

    def test_not_reexported_modules_are_importable(self):
        import importlib

        for module_path in NOT_REEXPORTED:
            importlib.import_module(module_path)

    def test_not_reexported_modules_are_not_in_all(self):
        """These are documented as import-directly; listing them in __all__ would contradict that."""
        for module_path in NOT_REEXPORTED:
            leaf = module_path.split(".")[-1]
            assert leaf not in dyf.__all__


class TestOverview:
    def test_default_lists_every_group(self):
        text = overview()
        for name in API_GROUPS:
            assert name in text

    def test_reports_the_real_total(self):
        assert str(len(dyf.__all__)) in overview()

    def test_group_argument_lists_that_group_s_names(self):
        text = overview("dedup")
        assert "near_duplicate_clusters" in text
        assert "BridgeIndex" not in text, "should show only the requested group"

    def test_as_dict_is_parseable(self):
        data = overview(as_dict=True)
        assert set(data) == set(API_GROUPS)
        assert data["trees"]["start_here"] == "build_dyf_tree"
        assert isinstance(data["trees"]["names"], list)

    def test_unknown_group_names_the_valid_ones(self):
        import pytest

        with pytest.raises(KeyError, match="dedup"):
            overview("no-such-group")

    def test_exposed_on_the_package(self):
        assert dyf.overview is overview
