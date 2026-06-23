import math

from quant_master.contrib.strategy.order_generator import _finite_positive_amounts


def test_finite_positive_amounts_drops_nonfinite_and_nonpositive_targets():
    filtered = _finite_positive_amounts(
        {
            "A": 100.0,
            "B": float("nan"),
            "C": float("inf"),
            "D": 0.0,
            "E": -10.0,
        }
    )

    assert filtered == {"A": 100.0}
    assert all(math.isfinite(v) and v > 0 for v in filtered.values())
