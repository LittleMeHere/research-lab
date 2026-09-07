from usage.rates import unit_price


def test_unit_price_known_metric():
    assert unit_price("api_calls") == 0.002


def test_unit_price_unknown_metric_is_free():
    assert unit_price("beta_feature") == 0.0
