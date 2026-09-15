import pytest

from quant_master.contrib.data.feature_registry import FeatureRegistry, FeatureSpec


def make_feature(name, family, descriptor, window, requirements=("close",), lag=0, source="tdx"):
    return FeatureSpec(
        name=name,
        family=family,
        descriptor=descriptor,
        window=window,
        data_requirements=requirements,
        available_lag=lag,
        source=source,
    )


def test_registry_expands_columns_and_preserves_metadata_order():
    registry = FeatureRegistry(
        [
            make_feature("MOM5", "momentum", "$close/Ref($close, 5)-1", 5),
            make_feature(
                "AMOUNT_REL20",
                "liquidity",
                "$amount/(Mean($amount, 20)+1e-12)-1",
                20,
                requirements=("amount",),
                lag=1,
                source="exchange_eod",
            ),
            make_feature("MOM20", "momentum", "$close/Ref($close, 20)-1", 20),
        ]
    )

    fields, names = registry.expand_columns()

    assert names == ["MOM5", "AMOUNT_REL20", "MOM20"]
    assert fields == [
        "$close/Ref($close, 5)-1",
        "$amount/(Mean($amount, 20)+1e-12)-1",
        "$close/Ref($close, 20)-1",
    ]
    assert registry.families == ("momentum", "liquidity")
    assert registry.family_counts() == {"momentum": 2, "liquidity": 1}
    assert registry.get("AMOUNT_REL20").available_lag == 1


def test_registry_supports_family_selection_and_ablation():
    registry = FeatureRegistry(
        [
            make_feature("MOM5", "momentum", "MOMENTUM_EXPR", 5),
            make_feature("LIQ20", "liquidity", "LIQUIDITY_EXPR", 20),
            make_feature("RISK5_20", "risk", "RISK_EXPR", [5, 20]),
        ]
    )

    assert registry.expand_columns(families=("momentum", "risk"))[1] == ["MOM5", "RISK5_20"]
    assert registry.expand_columns(exclude_families=("liquidity",))[1] == ["MOM5", "RISK5_20"]

    ablated = registry.ablate("momentum")
    ablated.register(make_feature("LIQ60", "liquidity", "LIQUIDITY_60_EXPR", 60))

    assert ablated.expand_columns()[1] == ["LIQ20", "RISK5_20", "LIQ60"]
    assert registry.expand_columns()[1] == ["MOM5", "LIQ20", "RISK5_20"]
    assert registry.get("RISK5_20").window == (5, 20)


def test_duplicate_registration_is_rejected_atomically():
    registry = FeatureRegistry([make_feature("MOM5", "momentum", "MOMENTUM_EXPR", 5)])

    with pytest.raises(ValueError, match="duplicate feature name.*MOM5"):
        registry.register(make_feature("MOM5", "other", "OTHER_EXPR", 10))

    with pytest.raises(ValueError, match="duplicate feature name.*LIQ20"):
        registry.register_many(
            [
                make_feature("LIQ20", "liquidity", "LIQUIDITY_EXPR", 20),
                make_feature("LIQ20", "risk", "RISK_EXPR", 20),
            ]
        )

    assert registry.expand_columns()[1] == ["MOM5"]


@pytest.mark.parametrize(
    "overrides, error_type, match",
    [
        ({"family": ""}, ValueError, "family"),
        ({"descriptor": " "}, ValueError, "descriptor"),
        ({"window": -1}, ValueError, "window"),
        ({"window": (5, "20")}, TypeError, "window values"),
        ({"data_requirements": ()}, ValueError, "data_requirements"),
        ({"data_requirements": "close"}, TypeError, "not a string"),
        ({"data_requirements": ("close", "close")}, ValueError, "duplicates"),
        ({"available_lag": -1}, ValueError, "available_lag"),
    ],
)
def test_feature_spec_validates_metadata(overrides, error_type, match):
    values = {
        "name": "MOM5",
        "family": "momentum",
        "descriptor": "MOMENTUM_EXPR",
        "window": 5,
        "data_requirements": ("close",),
        "available_lag": 0,
        "source": "tdx",
    }
    values.update(overrides)

    with pytest.raises(error_type, match=match):
        FeatureSpec(**values)


def test_register_many_only_accepts_feature_specs():
    registry = FeatureRegistry()

    with pytest.raises(TypeError, match="FeatureSpec"):
        registry.register_many([make_feature("MOM5", "momentum", "MOMENTUM_EXPR", 5), object()])

    assert len(registry) == 0
