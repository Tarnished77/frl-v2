"""Finance-native agent-based survival dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .config import ModelConfig
from .rng import make_random_inputs


@dataclass(frozen=True, order=True)
class Condition:
    calibration_ratio: float
    liquidity_intensity: int
    market_intensity: int
    population_design: str = "heterogeneous"

    def validate(self) -> None:
        if self.calibration_ratio <= 0:
            raise ValueError("calibration_ratio must be positive")
        if self.liquidity_intensity not in (0, 1, 2):
            raise ValueError("liquidity_intensity must be 0, 1, or 2")
        if self.market_intensity not in (0, 1, 2):
            raise ValueError("market_intensity must be 0, 1, or 2")
        if self.population_design not in ("heterogeneous", "homogeneous"):
            raise ValueError("population_design must be heterogeneous or homogeneous")

    @property
    def condition_id(self) -> str:
        rho = str(self.calibration_ratio).replace(".", "p")
        return (
            f"rho_{rho}_liq_{self.liquidity_intensity}_market_{self.market_intensity}_"
            f"{self.population_design}"
        )


@dataclass(frozen=True)
class SimulationResult:
    agent_rows: tuple[dict[str, Any], ...]
    daily_rows: tuple[dict[str, Any], ...]
    run_row: dict[str, Any]


def channel_budgets(stress_budget: float, calibration_ratio: float) -> tuple[float, float]:
    if stress_budget <= 0 or calibration_ratio <= 0:
        raise ValueError("stress_budget and calibration_ratio must be positive")
    liquidity_budget = stress_budget * calibration_ratio / (1.0 + calibration_ratio)
    market_budget = stress_budget / (1.0 + calibration_ratio)
    return liquidity_budget, market_budget


def market_loss_fraction(
    config: ModelConfig,
    market_budget: float,
    market_intensity: int,
) -> float:
    if market_intensity == 0:
        return 0.0
    loss = (
        market_budget
        * market_intensity
        / (config.market_event_probability * config.reference_risky_value)
    )
    if loss > config.maximum_market_loss_fraction + config.exit_tolerance:
        raise ValueError(
            f"market loss fraction {loss:.6f} exceeds configured maximum "
            f"{config.maximum_market_loss_fraction:.6f}"
        )
    return loss


def _initial_state(
    config: ModelConfig,
    namespace: str,
    replication_id: str,
    population_design: str,
) -> tuple[np.ndarray, ...]:
    random_inputs = make_random_inputs(config, namespace, replication_id)
    population = config.population

    if population_design == "homogeneous":
        leverage = np.full(population, config.reference_leverage, dtype=float)
        cash_share = np.full(population, config.reference_cash_share, dtype=float)
        payment_multiplier = np.ones(population, dtype=float)
    else:
        leverage = config.leverage_low + random_inputs.leverage_u * (
            config.leverage_high - config.leverage_low
        )
        cash_share = config.cash_share_low + random_inputs.cash_share_u * (
            config.cash_share_high - config.cash_share_low
        )
        payment_multiplier = config.payment_multiplier_low + random_inputs.payment_multiplier_u * (
            config.payment_multiplier_high - config.payment_multiplier_low
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
        payment_multiplier,
        random_inputs.liquidity_z,
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
        payment_multiplier,
        liquidity_z,
        market_event_u,
    ) = _initial_state(config, namespace, replication_id, condition.population_design)

    initial_cash = cash.copy()
    initial_risky_value = units.copy()
    initial_liabilities = liabilities.copy()
    population = config.population
    alive = np.ones(population, dtype=bool)
    exit_day = np.zeros(population, dtype=int)
    exit_reason = np.full(population, "censored", dtype=object)
    total_paid = np.zeros(population, dtype=float)
    total_unpaid = np.zeros(population, dtype=float)
    total_forced_sale = np.zeros(population, dtype=float)
    exit_cash = np.full(population, np.nan, dtype=float)
    exit_risky_value = np.full(population, np.nan, dtype=float)
    exit_liabilities = np.full(population, np.nan, dtype=float)
    exit_equity = np.full(population, np.nan, dtype=float)

    liquidity_budget, market_budget = channel_budgets(
        stress_budget, condition.calibration_ratio
    )
    adverse_loss_fraction = market_loss_fraction(
        config, market_budget, condition.market_intensity
    )
    price = 1.0
    daily_rows: list[dict[str, Any]] = []
    run_id = f"{condition.condition_id}__{replication_id}"

    for day_index in range(config.horizon):
        day = day_index + 1
        at_risk_start = int(alive.sum())
        market_event = bool(
            condition.market_intensity > 0
            and market_event_u[day_index] < config.market_event_probability
        )
        market_loss = adverse_loss_fraction if market_event else 0.0
        if market_event:
            price *= 1.0 - market_loss

        noise = np.exp(
            config.liquidity_noise_sigma * liquidity_z[day_index]
            - 0.5 * config.liquidity_noise_sigma**2
        )
        obligation = np.zeros(population, dtype=float)
        obligation[alive] = (
            liquidity_budget
            * condition.liquidity_intensity
            * payment_multiplier[alive]
            * noise[alive]
        )

        initial_payment = np.minimum(cash, obligation)
        initial_payment[~alive] = 0.0
        cash -= initial_payment
        gap = obligation - initial_payment

        requested_units = np.zeros(population, dtype=float)
        can_sell = alive & (gap > config.exit_tolerance) & (units > config.exit_tolerance)
        requested_units[can_sell] = np.minimum(gap[can_sell] / price, units[can_sell])
        preimpact_sale_value = float(np.sum(requested_units * price))
        active_risky_value = float(np.sum(units[alive] * price))
        sale_fraction = (
            preimpact_sale_value / active_risky_value
            if active_risky_value > config.exit_tolerance
            else 0.0
        )
        price_impact_loss = 1.0 - math.exp(-config.price_impact_lambda * sale_fraction)
        if preimpact_sale_value > config.exit_tolerance:
            price *= 1.0 - price_impact_loss

        sale_proceeds = requested_units * price
        units -= requested_units
        cash += sale_proceeds
        forced_sale_payment = np.minimum(cash, gap)
        forced_sale_payment[~alive] = 0.0
        cash -= forced_sale_payment
        unpaid = gap - forced_sale_payment

        total_paid += initial_payment + forced_sale_payment
        total_unpaid += unpaid
        total_forced_sale += sale_proceeds

        equity = cash + units * price - liabilities
        liquidity_failure = alive & (unpaid > config.exit_tolerance)
        insolvency = alive & (equity <= config.exit_tolerance)
        exiting = liquidity_failure | insolvency
        both = liquidity_failure & insolvency
        exit_reason[exiting & ~both & liquidity_failure] = "unpaid_liquidity"
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
                "population_design": condition.population_design,
                "calibration_ratio": condition.calibration_ratio,
                "liquidity_intensity": condition.liquidity_intensity,
                "market_intensity": condition.market_intensity,
                "day": day,
                "at_risk_start": at_risk_start,
                "at_risk_end": int(alive.sum()),
                "exit_count": int(exiting.sum()),
                "liquidity_exit_count": int((liquidity_failure & ~insolvency).sum()),
                "insolvency_exit_count": int((insolvency & ~liquidity_failure).sum()),
                "joint_exit_count": int(both.sum()),
                "market_event": int(market_event),
                "market_loss_fraction": market_loss,
                "price_impact_loss": price_impact_loss,
                "asset_price": price,
                "preimpact_forced_sale_value": preimpact_sale_value,
                "forced_sale_proceeds": float(np.sum(sale_proceeds)),
                "total_obligation": float(np.sum(obligation)),
                "total_unpaid": float(np.sum(unpaid)),
                "mean_active_equity": float(np.mean(active_equity)) if active_equity.size else math.nan,
                "mean_active_cash": float(np.mean(active_cash)) if active_cash.size else math.nan,
            }
        )

    current_risky_value = units * price
    current_equity = cash + current_risky_value - liabilities
    terminal_cash = np.where(alive, cash, exit_cash)
    terminal_risky_value = np.where(alive, current_risky_value, exit_risky_value)
    terminal_liabilities = np.where(alive, liabilities, exit_liabilities)
    terminal_equity = np.where(alive, current_equity, exit_equity)
    agent_rows: list[dict[str, Any]] = []
    for agent_id in range(population):
        event = int(exit_day[agent_id] > 0)
        survival_time = int(exit_day[agent_id] if event else config.horizon)
        agent_rows.append(
            {
                "run_id": run_id,
                "replication_id": replication_id,
                "condition_id": condition.condition_id,
                "population_design": condition.population_design,
                "calibration_ratio": condition.calibration_ratio,
                "liquidity_intensity": condition.liquidity_intensity,
                "market_intensity": condition.market_intensity,
                "agent_id": agent_id,
                "event": event,
                "exit_day": int(exit_day[agent_id]),
                "survival_time": survival_time,
                "exit_reason": str(exit_reason[agent_id]),
                "initial_equity": config.initial_equity,
                "initial_leverage": float(leverage[agent_id]),
                "initial_cash_share": float(cash_share[agent_id]),
                "initial_cash": float(initial_cash[agent_id]),
                "initial_risky_value": float(initial_risky_value[agent_id]),
                "initial_liabilities": float(initial_liabilities[agent_id]),
                "final_cash": float(terminal_cash[agent_id]),
                "final_risky_value": float(terminal_risky_value[agent_id]),
                "final_liabilities": float(terminal_liabilities[agent_id]),
                "final_equity": float(terminal_equity[agent_id]),
                "total_payment": float(total_paid[agent_id]),
                "total_unpaid": float(total_unpaid[agent_id]),
                "total_forced_sale_proceeds": float(total_forced_sale[agent_id]),
            }
        )

    event_count = int(np.count_nonzero(exit_day))
    run_row = {
        "run_id": run_id,
        "replication_id": replication_id,
        "condition_id": condition.condition_id,
        "population_design": condition.population_design,
        "calibration_ratio": condition.calibration_ratio,
        "liquidity_intensity": condition.liquidity_intensity,
        "market_intensity": condition.market_intensity,
        "population": population,
        "horizon": config.horizon,
        "stress_budget": stress_budget,
        "liquidity_budget": liquidity_budget,
        "market_budget": market_budget,
        "market_event_loss_fraction": adverse_loss_fraction,
        "event_count": event_count,
        "exit_rate": event_count / population,
        "rmst_days": float(np.mean(np.where(exit_day > 0, exit_day, config.horizon))),
        "final_asset_price": price,
        "mean_final_equity": float(np.mean(terminal_equity)),
        "mean_final_cash": float(np.mean(terminal_cash)),
        "mean_total_unpaid": float(np.mean(total_unpaid)),
        "mean_forced_sale_proceeds": float(np.mean(total_forced_sale)),
    }

    return SimulationResult(tuple(agent_rows), tuple(daily_rows), run_row)
