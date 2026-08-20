def clamp(value, lo, hi):
    """Clamp `value` into the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))
