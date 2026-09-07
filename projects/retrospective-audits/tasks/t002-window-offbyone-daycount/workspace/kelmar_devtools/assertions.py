def approx_money(actual, expected, tolerance=0.005):
    return abs(actual - expected) <= tolerance
