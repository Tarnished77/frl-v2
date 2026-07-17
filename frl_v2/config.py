"""Configuration loading and validation for the FRL v2 model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    model_version: str
    horizon: int
    population: int
    initial_equity: float
    reference_leverage: float
    leverage_low: float
    leverage_high: float
    reference_cash_share: float
    cash_share_low: float
    cash_share_high: float
    payment_multiplier_low: float
    payment_multiplier_high: float
    liquidity_noise_sigma: float
    market_event_probability: float
    price_impact_lambda: float
    maximum_market_loss_fraction: float
    exit_tolerance: float

    @property
    def reference_assets(self) -> float:
        return self.initial_equity * self.reference_leverage

    @property
    def reference_risky_value(self) -> float:
        return self.reference_assets * (1.0 - self.reference_cash_share)

    def validate(self) -> None:
        if not self.model_version:
            raise ValueError("model_version must be non-empty")
        if self.horizon < 1 or self.population < 1:
            raise ValueError("horizon and population must be positive")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not (1.0 <= self.leverage_low <= self.reference_leverage <= self.leverage_high):
            raise ValueError("leverage bounds must contain the reference leverage")
        if not (0.0 < self.cash_share_low <= self.reference_cash_share <= self.cash_share_high < 1.0):
            raise ValueError("cash-share bounds must lie in (0, 1)")
        if not (0.0 < self.payment_multiplier_low <= 1.0 <= self.payment_multiplier_high):
            raise ValueError("payment-multiplier bounds must contain one")
        if self.liquidity_noise_sigma < 0:
            raise ValueError("liquidity_noise_sigma cannot be negative")
        if not (0.0 < self.market_event_probability < 1.0):
            raise ValueError("market_event_probability must lie in (0, 1)")
        if self.price_impact_lambda < 0:
            raise ValueError("price_impact_lambda cannot be negative")
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
    liquidity_intensities: tuple[int, ...]
    market_intensities: tuple[int, ...]
    primary_replications: int
    homogeneous_replications: int
    finite_size_replications: int
    finite_size_populations: tuple[int, ...]
    pilot_replications: int
    pilot_stress_budgets: tuple[float, ...]
    pilot_target_exit_rate: float
    pilot_acceptable_exit_rate_low: float
    pilot_acceptable_exit_rate_high: float
    finite_size_exit_risk_tolerance: float
    finite_size_rmst_tolerance_days: float
    bootstrap_replications: int

    def validate(self) -> None:
        if not self.experiment_namespace:
            raise ValueError("experiment_namespace must be non-empty")
        if self.calibration_ratios != (0.5, 1.0, 2.0):
            raise ValueError("the locked calibration ratios are 0.5, 1.0, and 2.0")
        if self.liquidity_intensities != (0, 1, 2):
            raise ValueError("the locked liquidity intensities are 0, 1, and 2")
        if self.market_intensities != (0, 1, 2):
            raise ValueError("the locked market intensities are 0, 1, and 2")
        positive_counts = (
            self.primary_replications,
            self.homogeneous_replications,
            self.finite_size_replications,
            self.pilot_replications,
            self.bootstrap_replications,
        )
        if any(value < 1 for value in positive_counts):
            raise ValueError("replication counts must be positive")
        if any(value < 1 for value in self.finite_size_populations):
            raise ValueError("finite-size populations must be positive")
        if any(value <= 0 for value in self.pilot_stress_budgets):
            raise ValueError("pilot stress budgets must be positive")
        if not (
            0.0
            <= self.pilot_acceptable_exit_rate_low
            < self.pilot_target_exit_rate
            < self.pilot_acceptable_exit_rate_high
            <= 1.0
        ):
            raise ValueError("pilot exit-rate targets are not ordered")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in data.items()}


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
        "liquidity_intensities",
        "market_intensities",
        "finite_size_populations",
        "pilot_stress_budgets",
    ):
        data[key] = tuple(data[key])
    config = DesignConfig(**data)
    config.validate()
    return config
