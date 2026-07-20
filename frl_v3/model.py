"""Leveraged-institution funding, market-loss, and fire-sale dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .config import ModelConfig
from .rng import make_random_inputs


CASH_SCENARIOS = ("baseline", "low", "reference", "high")


def _token(value: float) -> str:
    return format(value, ".6g").replace(".", "p")


@dataclass(frozen=True, order=True)
class Condition:
    calibration_ratio: float
    funding_intensity: int
    market_intensity: int
    price_impact_lambda: float
    market_event_probability: float
    cash_scenario: str = "baseline"

    def validate(self) -> None:
        if self.calibration_ratio <= 0:
            raise ValueError("calibration_ratio must be positive")
        if self.funding_intensity not in (0, 1, 2):
            raise ValueError("funding_intensity must be 0, 1, or 2")
        if self.market_intensity not in (0, 1, 2):
            raise ValueError("market_intensity must be 0, 1, or 2")
        if self.price_impact_lambda < 0:
            raise ValueError("price_impact_lambda cannot be negative")
        if not 0.0 < self.market_event_probability < 1.0:
            raise ValueError("market_event_probability must lie in (0, 1)")
        if self.cash_scenario not in CASH_SCENARIOS:
            raise ValueError(f"Unknown cash scenario: {self.cash_scenario}")

    @property
    def feedback_state(self) -> str:
        return "none" if self.price_impact_lambda == 0 else "active"

    @property
    def condition_id(self) -> str:
        return (
            f"rho_{_token(self.calibration_ratio)}_fund_{self.funding_intensity}_"
            f"market_{self.market_intensity}_lambda_{_token(self.price_impact_lambda)}_"
            f"p_{_token(self.market_event_probability)}_cash_{self.cash_scenario}"
        )


@dataclass(frozen=True)
class SimulationResult:
    agent_rows: tuple[dict[str, Any], ...]
    daily_rows: tuple[dict[str, Any], ...]
    run_row: dict[str, Any]


@dataclass(frozen=True)
class FundingSettlement:
    cash: np.ndarray
    units: np.ndarray
    liabilities: np.ndarray
    price: float
    paid: np.ndarray
    unpaid: np.ndarray
    requested_units: np.ndarray
    sale_proceeds: np.ndarray
    preimpact_sale_value: float
    sale_fraction: float
    price_impact_loss: float


def channel_budgets(
    stress_budget: float, calibration_ratio: float
) -> tuple[float, float]:
    if stress_budget <= 0 or calibration_ratio <= 0:
        raise ValueError("stress_budget and calibration_ratio must be positive")
    funding_budget = stress_budget * calibration_ratio / (1.0 + calibration_ratio)
    market_budget = stress_budget / (1.0 + calibration_ratio)
    return funding_budget, market_budget


def market_loss_fraction(
    config: ModelConfig,
    market_budget: float,
    market_intensity: int,
    event_probability: float,
) -> float:
    if market_intensity == 0:
        return 0.0
    target = market_budget * market_intensity
    if not 0.0 < target < config.reference_risky_value:
        raise ValueError("market target must be below reference risky value")
    expected_price_ratio = 1.0 - target / config.reference_risky_value
    daily_expected_ratio = expected_price_ratio ** (1.0 / config.horizon)
    loss = (1.0 - daily_expected_ratio) / event_probability
    if loss > config.maximum_market_loss_fraction + config.exit_tolerance:
        raise ValueError(
            f"market loss fraction {loss:.6f} exceeds configured maximum "
            f"{config.maximum_market_loss_fraction:.6f}"
        )
    return loss


def expected_reference_market_loss(
    config: ModelConfig, event_probability: float, event_loss_fraction: float
) -> float:
    price_ratio = (1.0 - event_probability * event_loss_fraction) ** config.horizon
    return config.reference_risky_value * (1.0 - price_ratio)


def daily_reference_funding_withdrawal(
    config: ModelConfig, funding_budget: float, funding_intensity: int
) -> float:
    return funding_budget * funding_intensity / config.horizon


def settle_funding_withdrawal(
    cash: np.ndarray,
    units: np.ndarray,
    liabilities: np.ndarray,
    alive: np.ndarray,
    price: float,
    withdrawal: np.ndarray,
    price_impact_lambda: float,
    tolerance: float,
) -> FundingSettlement:
    cash = cash.copy()
    units = units.copy()
    liabilities = liabilities.copy()
    withdrawal = np.minimum(np.maximum(withdrawal, 0.0), liabilities)
    withdrawal[~alive] = 0.0

    cash_payment = np.minimum(cash, withdrawal)
    cash_payment[~alive] = 0.0
    cash -= cash_payment
    gap = withdrawal - cash_payment

    can_sell = alive & (gap > tolerance) & (units > tolerance)
    active_units = float(np.sum(units[alive]))
    if not np.any(can_sell) or active_units <= tolerance:
        sale_fraction = 0.0
    else:
        def implied_sale_fraction(candidate: float) -> float:
            candidate_price = price * math.exp(
                -price_impact_lambda * candidate
            )
            requested = np.zeros_like(units)
            requested[can_sell] = np.minimum(
                gap[can_sell] / candidate_price, units[can_sell]
            )
            return float(np.sum(requested) / active_units)

        lower = 0.0
        upper = 1.0
        if implied_sale_fraction(lower) <= tolerance:
            sale_fraction = 0.0
        else:
            for _ in range(100):
                midpoint = (lower + upper) / 2.0
                if implied_sale_fraction(midpoint) > midpoint:
                    lower = midpoint
                else:
                    upper = midpoint
            sale_fraction = (lower + upper) / 2.0

    price_impact_loss = 1.0 - math.exp(-price_impact_lambda * sale_fraction)
    postimpact_price = price * (1.0 - price_impact_loss)
    requested_units = np.zeros_like(units)
    requested_units[can_sell] = np.minimum(
        gap[can_sell] / postimpact_price, units[can_sell]
    )
    preimpact_sale_value = float(np.sum(requested_units * price))

    sale_proceeds = requested_units * postimpact_price
    units -= requested_units
    cash += sale_proceeds
    sale_payment = np.minimum(cash, gap)
    sale_payment[~alive] = 0.0
    cash -= sale_payment

    paid = cash_payment + sale_payment
    unpaid = withdrawal - paid
    liabilities -= paid
    liabilities = np.maximum(liabilities, 0.0)
    return FundingSettlement(
        cash=cash,
        units=units,
        liabilities=liabilities,
        price=postimpact_price,
        paid=paid,
        unpaid=unpaid,
        requested_units=requested_units,
        sale_proceeds=sale_proceeds,
        preimpact_sale_value=preimpact_sale_value,
        sale_fraction=sale_fraction,
        price_impact_loss=price_impact_loss,
    )


def _initial_state(
    config: ModelConfig,
    namespace: str,
    replication_id: str,
    cash_scenario: str,
) -> tuple[np.ndarray, ...]:
    random_inputs = make_random_inputs(config, namespace, replication_id)
    population = config.population
    leverage = config.leverage_low + random_inputs.leverage_u * (
        config.leverage_high - config.leverage_low
    )
    if cash_scenario == "baseline":
        cash_share = config.cash_share_low + random_inputs.cash_share_u * (
            config.cash_share_high - config.cash_share_low
        )
    else:
        selected = {
            "low": config.cash_share_low,
            "reference": config.reference_cash_share,
            "high": config.cash_share_high,
        }[cash_scenario]
        cash_share = np.full(population, selected, dtype=float)
    funding_multiplier = config.funding_multiplier_low + (
        random_inputs.funding_multiplier_u
        * (config.funding_multiplier_high - config.funding_multiplier_low)
    )
    assets = config.initial_equity * leverage
    cash = assets * cash_share
    risky_value = assets - cash
    units = risky_value.copy()
    liabilities = assets - config.initial_equity
    return (
        cash,
        units,
        liabilities,
        leverage,
        cash_share,
        funding_multiplier,
        random_inputs.funding_z,
        random_inputs.market_event_u,
    )


def simulate_run(
    config: ModelConfig,
    condition: Condition,
    stress_budget: float,
    namespace: str,
    replication_id: str,
) -> SimulationResult:
    config.validate()
    condition.validate()
    (
        cash,
        units,
        liabilities,
        leverage,
        cash_share,
        funding_multiplier,
        funding_z,
        market_event_u,
    ) = _initial_state(config, namespace, replication_id, condition.cash_scenario)

    initial_cash = cash.copy()
    initial_risky_value = units.copy()
    initial_liabilities = liabilities.copy()
    population = config.population
    alive = np.ones(population, dtype=bool)
    exit_day = np.zeros(population, dtype=int)
    exit_reason = np.full(population, "censored", dtype=object)
    total_requested = np.zeros(population, dtype=float)
    total_repaid = np.zeros(population, dtype=float)
    total_unpaid = np.zeros(population, dtype=float)
    total_forced_sale = np.zeros(population, dtype=float)
    total_market_value_loss = np.zeros(population, dtype=float)
    total_fire_sale_value_loss = np.zeros(population, dtype=float)
    exit_cash = np.full(population, np.nan, dtype=float)
    exit_risky_value = np.full(population, np.nan, dtype=float)
    exit_liabilities = np.full(population, np.nan, dtype=float)
    exit_equity = np.full(population, np.nan, dtype=float)

    funding_budget, market_budget = channel_budgets(
        stress_budget, condition.calibration_ratio
    )
    adverse_loss_fraction = market_loss_fraction(
        config,
        market_budget,
        condition.market_intensity,
        condition.market_event_probability,
    )
    reference_daily_withdrawal = daily_reference_funding_withdrawal(
        config, funding_budget, condition.funding_intensity
    )
    price = 1.0
    daily_rows: list[dict[str, Any]] = []
    run_id = f"{condition.condition_id}__{replication_id}"

    for day_index in range(config.horizon):
        day = day_index + 1
        at_risk_start = int(alive.sum())
        market_event = bool(
            condition.market_intensity > 0
            and market_event_u[day_index] < condition.market_event_probability
        )
        market_loss = adverse_loss_fraction if market_event else 0.0
        if market_event:
            total_market_value_loss[alive] += (
                units[alive] * price * market_loss
            )
            price *= 1.0 - market_loss

        noise = np.exp(
            config.funding_noise_sigma * funding_z[day_index]
            - 0.5 * config.funding_noise_sigma**2
        )
        withdrawal = np.zeros(population, dtype=float)
        withdrawal[alive] = (
            reference_daily_withdrawal
            * funding_multiplier[alive]
            * noise[alive]
        )
        withdrawal = np.minimum(withdrawal, liabilities)

        settlement = settle_funding_withdrawal(
            cash,
            units,
            liabilities,
            alive,
            price,
            withdrawal,
            condition.price_impact_lambda,
            config.exit_tolerance,
        )
        total_fire_sale_value_loss[alive] += (
            units[alive] * price * settlement.price_impact_loss
        )
        cash = settlement.cash
        units = settlement.units
        liabilities = settlement.liabilities
        price = settlement.price
        total_requested += withdrawal
        total_repaid += settlement.paid
        total_unpaid += settlement.unpaid
        total_forced_sale += settlement.sale_proceeds

        equity = cash + units * price - liabilities
        funding_failure = alive & (settlement.unpaid > config.exit_tolerance)
        insolvency = alive & (equity <= config.exit_tolerance)
        exiting = funding_failure | insolvency
        both = funding_failure & insolvency
        exit_reason[exiting & ~both & funding_failure] = "funding_failure"
        exit_reason[exiting & ~both & insolvency] = "insolvency"
        exit_reason[both] = "both"
        exit_day[exiting] = day
        exit_cash[exiting] = cash[exiting]
        exit_risky_value[exiting] = units[exiting] * price
        exit_liabilities[exiting] = liabilities[exiting]
        exit_equity[exiting] = equity[exiting]
        alive[exiting] = False

        active_equity = equity[alive]
        active_cash = cash[alive]
        daily_rows.append(
            {
                "run_id": run_id,
                "replication_id": replication_id,
                "condition_id": condition.condition_id,
                "calibration_ratio": condition.calibration_ratio,
                "funding_intensity": condition.funding_intensity,
                "market_intensity": condition.market_intensity,
                "price_impact_lambda": condition.price_impact_lambda,
                "market_event_probability": condition.market_event_probability,
                "cash_scenario": condition.cash_scenario,
                "day": day,
                "at_risk_start": at_risk_start,
                "at_risk_end": int(alive.sum()),
                "exit_count": int(exiting.sum()),
                "funding_exit_count": int((funding_failure & ~both).sum()),
                "insolvency_exit_count": int((insolvency & ~both).sum()),
                "joint_exit_count": int(both.sum()),
                "market_event": int(market_event),
                "market_loss_fraction": market_loss,
                "price_impact_loss": settlement.price_impact_loss,
                "forced_sale_fraction": settlement.sale_fraction,
                "asset_price": price,
                "preimpact_forced_sale_value": settlement.preimpact_sale_value,
                "forced_sale_proceeds": float(settlement.sale_proceeds.sum()),
                "total_funding_requested": float(withdrawal.sum()),
                "total_funding_repaid": float(settlement.paid.sum()),
                "total_unpaid": float(settlement.unpaid.sum()),
                "total_liabilities": float(liabilities[alive].sum()),
                "mean_active_equity": (
                    float(np.mean(active_equity)) if active_equity.size else np.nan
                ),
                "mean_active_cash": (
                    float(np.mean(active_cash)) if active_cash.size else np.nan
                ),
            }
        )

    final_equity = cash + units * price - liabilities
    terminal_cash = np.where(np.isnan(exit_cash), cash, exit_cash)
    terminal_risky_value = np.where(
        np.isnan(exit_risky_value), units * price, exit_risky_value
    )
    terminal_liabilities = np.where(
        np.isnan(exit_liabilities), liabilities, exit_liabilities
    )
    terminal_equity = np.where(
        np.isnan(exit_equity), final_equity, exit_equity
    )
    agent_rows = []
    for index in range(population):
        event = int(exit_day[index] > 0)
        agent_rows.append(
            {
                "run_id": run_id,
                "replication_id": replication_id,
                "condition_id": condition.condition_id,
                "calibration_ratio": condition.calibration_ratio,
                "funding_intensity": condition.funding_intensity,
                "market_intensity": condition.market_intensity,
                "price_impact_lambda": condition.price_impact_lambda,
                "market_event_probability": condition.market_event_probability,
                "cash_scenario": condition.cash_scenario,
                "agent_id": index + 1,
                "event": event,
                "exit_day": int(exit_day[index]),
                "survival_time": (
                    int(exit_day[index]) if event else config.horizon
                ),
                "exit_reason": str(exit_reason[index]),
                "initial_equity": config.initial_equity,
                "initial_leverage": float(leverage[index]),
                "initial_cash_share": float(cash_share[index]),
                "initial_cash": float(initial_cash[index]),
                "initial_risky_value": float(initial_risky_value[index]),
                "initial_liabilities": float(initial_liabilities[index]),
                "final_cash": float(terminal_cash[index]),
                "final_risky_value": float(terminal_risky_value[index]),
                "final_liabilities": float(terminal_liabilities[index]),
                "final_equity": float(terminal_equity[index]),
                "total_funding_requested": float(total_requested[index]),
                "total_funding_repaid": float(total_repaid[index]),
                "total_unpaid": float(total_unpaid[index]),
                "total_forced_sale_proceeds": float(total_forced_sale[index]),
                "total_market_value_loss": float(
                    total_market_value_loss[index]
                ),
                "total_fire_sale_value_loss": float(
                    total_fire_sale_value_loss[index]
                ),
            }
        )

    event_count = int(np.count_nonzero(exit_day))
    equity_losses = config.initial_equity - terminal_equity
    tail_count = max(1, int(math.ceil(0.10 * population)))
    worst_tail_equity_loss = float(
        np.mean(np.sort(equity_losses)[-tail_count:])
    )
    run_row = {
        "run_id": run_id,
        "replication_id": replication_id,
        "condition_id": condition.condition_id,
        "calibration_ratio": condition.calibration_ratio,
        "funding_intensity": condition.funding_intensity,
        "market_intensity": condition.market_intensity,
        "price_impact_lambda": condition.price_impact_lambda,
        "market_event_probability": condition.market_event_probability,
        "cash_scenario": condition.cash_scenario,
        "population": population,
        "horizon": config.horizon,
        "stress_budget": stress_budget,
        "funding_budget": funding_budget,
        "market_budget": market_budget,
        "market_event_loss_fraction": adverse_loss_fraction,
        "target_cumulative_funding": (
            condition.funding_intensity * funding_budget
        ),
        "target_cumulative_market_loss": (
            condition.market_intensity * market_budget
        ),
        "event_count": event_count,
        "exit_rate": event_count / population,
        "rmst_days": float(
            np.mean(np.where(exit_day > 0, exit_day, config.horizon))
        ),
        "final_asset_price": price,
        "asset_price_loss": 1.0 - price,
        "mean_final_equity": float(np.mean(terminal_equity)),
        "mean_equity_loss": float(np.mean(equity_losses)),
        "p90_equity_loss": float(
            np.quantile(equity_losses, 0.90, method="linear")
        ),
        "worst_10pct_mean_equity_loss": worst_tail_equity_loss,
        "mean_final_cash": float(np.mean(terminal_cash)),
        "mean_final_liabilities": float(np.mean(terminal_liabilities)),
        "mean_liability_reduction": float(
            np.mean(initial_liabilities - terminal_liabilities)
        ),
        "mean_total_funding_requested": float(np.mean(total_requested)),
        "mean_total_funding_repaid": float(np.mean(total_repaid)),
        "mean_total_unpaid": float(np.mean(total_unpaid)),
        "mean_forced_sale_proceeds": float(np.mean(total_forced_sale)),
        "mean_market_value_loss": float(
            np.mean(total_market_value_loss)
        ),
        "mean_fire_sale_value_loss": float(
            np.mean(total_fire_sale_value_loss)
        ),
    }
    return SimulationResult(tuple(agent_rows), tuple(daily_rows), run_row)
