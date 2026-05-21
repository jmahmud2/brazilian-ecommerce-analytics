"""
Smoke tests — PROVIDED to candidates.

These verify your functions return the correct types and shapes.
Run with: pytest tests/test_smoke.py -v

Note: These tests require the dataset to be downloaded first.
Run `python data/download_data.py` before running tests.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestPart1:
    """Part 1: EDA & Business Overview"""

    def test_business_overview_returns_dict(self):
        from analysis import get_business_overview

        result = get_business_overview()
        assert isinstance(result, dict), "get_business_overview() must return a dict"

    def test_business_overview_has_required_keys(self):
        from analysis import get_business_overview

        result = get_business_overview()
        required_keys = {"total_revenue", "total_orders", "avg_order_value", "revenue_by_month"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_business_overview_types(self):
        from analysis import get_business_overview

        result = get_business_overview()
        assert isinstance(result["total_revenue"], (int, float))
        assert isinstance(result["total_orders"], (int, np.integer))
        assert isinstance(result["avg_order_value"], (int, float))
        assert isinstance(result["revenue_by_month"], pd.DataFrame)

    def test_revenue_by_month_shape(self):
        from analysis import get_business_overview

        result = get_business_overview()
        rbm = result["revenue_by_month"]
        assert "month" in rbm.columns, "revenue_by_month must have a 'month' column"
        assert "revenue" in rbm.columns, "revenue_by_month must have a 'revenue' column"
        assert len(rbm) > 0, "revenue_by_month must not be empty"


class TestPart2:
    """Part 2: Demand Forecasting"""

    def test_monthly_demand_returns_dataframe(self):
        from analysis import prepare_monthly_demand

        result = prepare_monthly_demand()
        assert isinstance(result, pd.DataFrame)

    def test_monthly_demand_columns(self):
        from analysis import prepare_monthly_demand

        result = prepare_monthly_demand()
        assert "month" in result.columns
        assert "order_count" in result.columns

    def test_monthly_demand_sorted(self):
        from analysis import prepare_monthly_demand

        result = prepare_monthly_demand()
        months = pd.to_datetime(result["month"])
        assert months.is_monotonic_increasing, "monthly demand must be sorted by month"

    def test_forecast_demand_returns_dict(self):
        from analysis import prepare_monthly_demand, forecast_demand
        from holdout_config import TRAIN_END_DATE, HOLDOUT_HORIZON

        demand = prepare_monthly_demand()
        train = demand[demand["month"] <= TRAIN_END_DATE]
        result = forecast_demand(train, HOLDOUT_HORIZON)
        assert isinstance(result, dict)

    def test_forecast_demand_has_required_keys(self):
        from analysis import prepare_monthly_demand, forecast_demand
        from holdout_config import TRAIN_END_DATE, HOLDOUT_HORIZON

        demand = prepare_monthly_demand()
        train = demand[demand["month"] <= TRAIN_END_DATE]
        result = forecast_demand(train, HOLDOUT_HORIZON)
        required_keys = {"model_name", "predictions", "rmse", "mae"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_forecast_predictions_length(self):
        from analysis import prepare_monthly_demand, forecast_demand
        from holdout_config import TRAIN_END_DATE, HOLDOUT_HORIZON

        demand = prepare_monthly_demand()
        train = demand[demand["month"] <= TRAIN_END_DATE]
        result = forecast_demand(train, HOLDOUT_HORIZON)
        assert len(result["predictions"]) == HOLDOUT_HORIZON, (
            f"predictions must have length {HOLDOUT_HORIZON}, got {len(result['predictions'])}"
        )


class TestPart3:
    """Part 3: LLM-Assisted Recommendations"""

    def test_analysis_context_returns_string(self):
        from analysis import build_analysis_context

        result = build_analysis_context()
        assert isinstance(result, str)
        assert len(result) > 100, "Analysis context should be a substantial summary"

    def test_recommendations_returns_dict(self):
        from analysis import generate_recommendations

        result = generate_recommendations()
        assert isinstance(result, dict)

    def test_recommendations_has_required_keys(self):
        from analysis import generate_recommendations

        result = generate_recommendations()
        required_keys = {"model_used", "prompt", "recommendations"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_recommendations_structure(self):
        from analysis import generate_recommendations

        result = generate_recommendations()
        recs = result["recommendations"]
        assert isinstance(recs, list)
        assert 3 <= len(recs) <= 5, f"Must have 3-5 recommendations, got {len(recs)}"

        for rec in recs:
            assert "action" in rec, "Each recommendation must have 'action'"
            assert "rationale" in rec, "Each recommendation must have 'rationale'"
            assert "priority" in rec, "Each recommendation must have 'priority'"
            assert rec["priority"] in ("high", "medium", "low"), (
                f"priority must be 'high', 'medium', or 'low', got '{rec['priority']}'"
            )