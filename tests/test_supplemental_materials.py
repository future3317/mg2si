from mg2si.data.supplemental_materials import normalize_supplemental_material_id


def test_layer_identity_is_preserved_for_supplemental_records():
    assert normalize_supplemental_material_id("MS-260526-SHS down 下层") == (
        "MS-260526-SHS down",
        "MS-260526-SHS",
        "down",
    )
    assert normalize_supplemental_material_id("MS-260110-SHS top") == (
        "MS-260110-SHS top",
        "MS-260110-SHS",
        "top",
    )
