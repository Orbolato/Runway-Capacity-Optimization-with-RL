import general_cap_optimization as run


def test_defaults_match_values_reported_in_section_4():
    assert run.PAPER_EPISODES == 2_000_000
    assert run.PAPER_STEP_SIZE == 0.5
    assert run.PAPER_DISCOUNT_RATE == 0.9
    assert run.PAPER_TRACE_DECAY == 0.5
    assert run.PAPER_INITIAL_EPSILON == 1.0
    assert run.PAPER_FINAL_EPSILON == 1e-6
    assert run.PAPER_EPSILON_DECAY_EPISODES == 1_000_000
