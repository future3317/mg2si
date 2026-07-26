from mg2si.mapping.resolver import load_aliases


def test_legacy_aliases_are_externalized():
    aliases = load_aliases()
    assert "MS-251016-SHS" in aliases["MS-251016"]

