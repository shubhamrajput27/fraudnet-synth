"""ml/validation/thresholds.py: CLAUDE.md D6 requires these frozen once any arm has been
evaluated (Phase 4+ has). This test is a regression guard against an accidental future edit, not
a test of arbitrary business logic — if it ever fails, that's a real D6 violation, not a false
positive to loosen."""
from ml.validation.thresholds import FROZEN_THRESHOLDS


def test_frozen_thresholds_match_the_d6_approved_values():
    assert FROZEN_THRESHOLDS.fidelity_floor == 0.5
    assert FROZEN_THRESHOLDS.novelty_floor == 0.95
    assert FROZEN_THRESHOLDS.mode_collapse_ceiling == 0.998
    assert FROZEN_THRESHOLDS.min_valid_rows_for_distributional_checks == 5
