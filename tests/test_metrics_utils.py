from openevolve.utils.metrics_utils import get_fitness_score, get_primary_metric_name


def test_primary_metric_priority_prefers_overall_score_before_average():
    metrics = {
        "overall_score": 0.9,
        "validity": 1.0,
        "raw_count": 20.0,
        "complexity": 3.0,
    }

    assert get_primary_metric_name(metrics) == "overall_score"
    assert get_fitness_score(metrics, feature_dimensions=["complexity"]) == 0.9


def test_primary_metric_priority_prefers_composite_score():
    metrics = {
        "composite_score": 0.42,
        "accuracy": 0.99,
    }

    assert get_primary_metric_name(metrics) == "composite_score"
    assert get_fitness_score(metrics) == 0.42


def test_combined_score_still_has_highest_priority():
    metrics = {
        "combined_score": 0.7,
        "overall_score": 0.9,
        "composite_score": 0.95,
    }

    assert get_primary_metric_name(metrics) == "combined_score"
    assert get_fitness_score(metrics) == 0.7


def test_nested_metrics_are_supported():
    metrics = {
        "metrics": {
            "combined_score": 0.25,
            "accuracy": 0.25,
        },
        "artifacts": {"status": "ok"},
    }

    assert get_primary_metric_name(metrics) == "combined_score"
    assert get_fitness_score(metrics) == 0.25
