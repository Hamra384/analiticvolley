def test_demo_ci_must_fail() -> None:
    """Demostración S1: CI debe fallar con un test roto. No mergear."""
    assert 1 + 1 == 3
