"""Configuration loading and validation for the FRL v3 model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    model_version: str
    anchor_version: str
    horizon: int
    population: int
    random_stream_population: int
    initial_equity: float
    reference_leverage: float
    leverage_low: float
    leverage_high: float
    reference_cash_share: float
    cash_share_low: float
    cash_share_high: float
    funding_multiplier_low: float
    funding_multiplier_high: float
    funding_noise_sigma: float
    market_event_probability: float
    benchmark_price_impact_lambda: float
    high_price_impact_lambda: float
    maximum_market_loss_fraction: float
    exit_tolerance: float

    @property
    def reference_assets(self) -> float:
        return self.initial_equity * self.reference_leverage

    @property
    def reference_risky_value(self) -> float:
        return self.reference_assets * (1.0 - self.reference_cash_share)

    @property
    def reference_liabilities(self) -> float:
        return self.reference_assets - self.initial_equity

    def validate(self) -> None:
        if self.model_version != "frl-v3.1":
            raise ValueError("model_version must be frl-v3.1")
        if not self.anchor_version:
            raise ValueError("anchor_version must be non-empty")
        if self.horizon < 1 or self.population < 1:
            raise ValueError("horizon and population must be positive")
        if self.population > self.random_stream_population:
            raise ValueError("population exceeds the locked random-stream population")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not (
            1.0 < self.leverage_low <= self.reference_leverage <= self.leverage_high
        ):
            raise ValueError("leverage bounds must contain the reference leverage")
        if not (
            0.0
            < self.cash_share_low
            <= self.reference_cash_share
            <= self.cash_share_high
            < 1.0
        ):
            raise ValueError("cash-share bounds must lie in (0, 1)")
        if not (
            0.0
            < self.funding_multiplier_low
            <= 1.0
            <= self.funding_multiplier_high
        ):
            raise ValueError("funding-multiplier bounds must contain one")
        if self.funding_noise_sigma < 0:
            raise ValueError("funding_noise_sigma cannot be negative")
        if not (0.0 < self.market_event_probability < 1.0):
            raise ValueError("market_event_probability must lie in (0, 1)")
        if not (
            0.0
            <= self.benchmark_price_impact_lambda
            <= self.high_price_impact_lambda
        ):
            raise ValueError("price-impact lambdas are not ordered")
        if not (0.0 < self.maximum_market_loss_fraction < 1.0):
            raise ValueError("maximum_market_loss_fraction must lie in (0, 1)")
        if self.exit_tolerance <= 0:
            raise ValueError("exit_tolerance must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DesignConfig:
    experiment_namespace: str
    calibration_ratios: tuple[float, ...]
    funding_intensities: tuple[int, ...]
    market_intensities: tuple[int, ...]
    primary_feedback_lambdas: tuple[float, ...]
    primary_replications: int
    maximum_primary_replications: int
    sensitivity_replications: int
    finite_size_replications: int
    finite_size_populations: tuple[int, ...]
    pilot_replications: int
    pilot_stress_budgets: tuple[float, ...]
    pilot_informative_exit_low: float
    pilot_informative_exit_high: float
    pilot_target_exit_rate: float
    pilot_min_informative_cells: int
    finite_size_exit_risk_tolerance: float
    finite_size_rmst_tolerance_days: float
    monte_carlo_exit_halfwidth_tolerance: float
    monte_carlo_rmst_halfwidth_tolerance: float
    bootstrap_replications: int
    market_frequency_sensitivity: tuple[float, ...]
    cash_buffer_sensitivity: tuple[str, ...]

    def validate(self) -> None:
        if not self.experiment_namespace:
            raise ValueError("experiment_namespace must be non-empty")
        if self.calibration_ratios != (0.5, 1.0, 2.0):
            raise ValueError("calibration ratios must be 0.5, 1.0, and 2.0")
        if self.funding_intensities != (0, 1, 2):
            raise ValueError("funding intensities must be 0, 1, and 2")
        if self.market_intensities != (0, 1, 2):
            raise ValueError("market intensities must be 0, 1, and 2")
        if self.primary_feedback_lambdas[0] != 0.0:
            raise ValueError("primary feedback design must include no-impact first")
        counts = (
            self.primary_replications,
            self.maximum_primary_replications,
            self.sensitivity_replications,
            self.finite_size_replications,
            self.pilot_replications,
            self.bootstrap_replications,
        )
        if any(value < 1 for value in counts):
            raise ValueError("replication counts must be positive")
        if self.maximum_primary_replications < self.primary_replications:
            raise ValueError("maximum primary replications is too small")
        if self.finite_size_populations != (50, 100, 200):
            raise ValueError("finite-size populations must be 50, 100, and 200")
        if tuple(sorted(self.pilot_stress_budgets)) != self.pilot_stress_budgets:
            raise ValueError("pilot budgets must be ordered")
        if any(value <= 0 for value in self.pilot_stress_budgets):
            raise ValueError("pilot budgets must be positive")
        if not (
            0.0
            <= self.pilot_informative_exit_low
            < self.pilot_target_exit_rate
            < self.pilot_informative_exit_high
            <= 1.0
        ):
            raise ValueError("pilot thresholds are not ordered")
        if self.pilot_min_informative_cells < 1:
            raise ValueError("pilot_min_informative_cells must be positive")
        if any(not 0.0 < value < 1.0 for value in self.market_frequency_sensitivity):
            raise ValueError("market-frequency sensitivities must lie in (0, 1)")
        if self.cash_buffer_sensitivity != ("low", "high"):
            raise ValueError("cash-buffer sensitivities must be low and high")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in data.items()
        }


@dataclass(frozen=True)
class ResilienceDesignConfig:
    design_version: str
    experiment_namespace: str
    primary_stress_budget: float
    stress_budget_sensitivity: tuple[float, ...]
    calibration_audit_budgets: tuple[float, ...]
    calibration_audit_replications: int
    calibration_ratios: tuple[float, ...]
    funding_intensities: tuple[int, ...]
    market_intensities: tuple[int, ...]
    primary_feedback_lambdas: tuple[float, ...]
    primary_replications: int
    maximum_primary_replications: int
    sensitivity_replications: int
    finite_size_replications: int
    finite_size_populations: tuple[int, ...]
    finite_size_mean_loss_tolerance: float
    finite_size_tail_loss_tolerance: float
    monte_carlo_mean_loss_halfwidth_tolerance: float
    monte_carlo_tail_loss_halfwidth_tolerance: float
    bootstrap_replications: int
    market_frequency_sensitivity: tuple[float, ...]
    cash_buffer_sensitivity: tuple[str, ...]

    def validate(self) -> None:
        if self.design_version != "frl-v3-resilience-1.0":
            raise ValueError("Unexpected resilience design version")
        if not self.experiment_namespace:
            raise ValueError("experiment_namespace must be non-empty")
        if self.primary_stress_budget != 0.15:
            raise ValueError("primary resilience stress budget must be 0.15")
        if self.stress_budget_sensitivity != (0.10, 0.30):
            raise ValueError("stress-budget sensitivities must be 0.10 and 0.30")
        if self.calibration_audit_budgets != (
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
        ):
            raise ValueError("calibration-audit budgets do not match the lock")
        if self.calibration_ratios != (0.5, 1.0, 2.0):
            raise ValueError("calibration ratios must be 0.5, 1.0, and 2.0")
        if self.funding_intensities != (0, 1, 2):
            raise ValueError("funding intensities must be 0, 1, and 2")
        if self.market_intensities != (0, 1, 2):
            raise ValueError("market intensities must be 0, 1, and 2")
        if self.primary_feedback_lambdas != (0.0, 0.15):
            raise ValueError("feedback lambdas must be 0 and 0.15")
        counts = (
            self.calibration_audit_replications,
            self.primary_replications,
            self.maximum_primary_replications,
            self.sensitivity_replications,
            self.finite_size_replications,
            self.bootstrap_replications,
        )
        if any(value < 1 for value in counts):
            raise ValueError("replication counts must be positive")
        if self.maximum_primary_replications <= self.primary_replications:
            raise ValueError("maximum replications must exceed initial blocks")
        if self.finite_size_populations != (50, 100, 200):
            raise ValueError("finite-size populations must be 50, 100, and 200")
        tolerances = (
            self.finite_size_mean_loss_tolerance,
            self.finite_size_tail_loss_tolerance,
            self.monte_carlo_mean_loss_halfwidth_tolerance,
            self.monte_carlo_tail_loss_halfwidth_tolerance,
        )
        if any(value <= 0 for value in tolerances):
            raise ValueError("resilience tolerances must be positive")
        if any(
            not 0.0 < value < 1.0
            for value in self.market_frequency_sensitivity
        ):
            raise ValueError("market frequencies must lie in (0, 1)")
        if self.cash_buffer_sensitivity != ("low", "high"):
            raise ValueError("cash-buffer sensitivities must be low and high")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in data.items()
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_config(path: Path) -> ModelConfig:
    config = ModelConfig(**_read_json(path))
    config.validate()
    return config


def load_design_config(path: Path) -> DesignConfig:
    data = _read_json(path)
    for key in (
        "calibration_ratios",
        "funding_intensities",
        "market_intensities",
        "primary_feedback_lambdas",
        "finite_size_populations",
        "pilot_stress_budgets",
        "market_frequency_sensitivity",
        "cash_buffer_sensitivity",
    ):
        data[key] = tuple(data[key])
    config = DesignConfig(**data)
    config.validate()
    return config


def load_resilience_design_config(path: Path) -> ResilienceDesignConfig:
    data = _read_json(path)
    for key in (
        "stress_budget_sensitivity",
        "calibration_audit_budgets",
        "calibration_ratios",
        "funding_intensities",
        "market_intensities",
        "primary_feedback_lambdas",
        "finite_size_populations",
        "market_frequency_sensitivity",
        "cash_buffer_sensitivity",
    ):
        data[key] = tuple(data[key])
    config = ResilienceDesignConfig(**data)
    config.validate()
    return config
