from experiments.psem_sortformer_adaptation_depth.preflight import static_checks


def test_static_contract_is_bound_to_authoritative_artifacts() -> None:
    checks = static_checks()
    assert checks
    assert all(row["passed"] for row in checks), checks
