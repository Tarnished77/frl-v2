from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
ANCHORS = ROOT / "anchors"
RAW = ANCHORS / "raw"
MANIFEST = ANCHORS / "source_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_values(payload: dict, mnemonics: list[str]) -> tuple[str, list[float]]:
    dates: set[str] = set()
    values: list[float] = []
    for mnemonic in mnemonics:
        observations = payload[mnemonic]["timeseries"]["aggregation"]
        date, value = observations[-1]
        dates.add(date)
        values.append(float(value))
    if len(dates) != 1:
        raise ValueError(f"latest observation dates do not match: {sorted(dates)}")
    return dates.pop(), values


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = math.floor((len(ordered) - 1) * probability)
    return ordered[index]


def fred_loss_summary(path: Path) -> dict[str, float | int | str]:
    observations: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row["SP500"]
            if not value or value == ".":
                continue
            observations.append((row["observation_date"], float(value)))

    window = 21
    losses = [
        max(0.0, 1.0 - observations[index][1] / observations[index - window][1])
        for index in range(window, len(observations))
    ]
    return {
        "start_date": observations[0][0],
        "end_date": observations[-1][0],
        "observations": len(observations),
        "windows": len(losses),
        "window_trading_days": window,
        "loss_q90": quantile(losses, 0.90),
        "loss_q95": quantile(losses, 0.95),
        "loss_q975": quantile(losses, 0.975),
        "loss_q99": quantile(losses, 0.99),
        "loss_max": max(losses),
    }


def rounded_outward(low: float, reference: float, high: float, digits: int) -> tuple:
    scale = 10**digits
    return (
        math.floor(low * scale) / scale,
        round(reference, digits),
        math.ceil(high * scale) / scale,
    )


def main() -> None:
    source_manifest = load_json(MANIFEST)
    for source in source_manifest["sources"]:
        raw_file = source.get("raw_file")
        expected = source.get("sha256")
        if not raw_file or not expected:
            continue
        actual = sha256(ANCHORS / raw_file)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {raw_file}: {actual}")

    leverage_payload = load_json(RAW / "ofr_leverage_by_strategy.json")
    comparable_leverage = [
        "FPF-STRATEGY_CREDIT_LEVERAGERATIO_GAVWMEAN",
        "FPF-STRATEGY_EQUITY_LEVERAGERATIO_GAVWMEAN",
        "FPF-STRATEGY_EVENT_LEVERAGERATIO_GAVWMEAN",
        "FPF-STRATEGY_OTHER_LEVERAGERATIO_GAVWMEAN",
        "FPF-STRATEGY_FOF_LEVERAGERATIO_GAVWMEAN",
    ]
    leverage_date, leverage_values = latest_values(
        leverage_payload, comparable_leverage
    )
    leverage_low, leverage_reference, leverage_high = rounded_outward(
        min(leverage_values), median(leverage_values), max(leverage_values), 1
    )

    cash_payload = load_json(RAW / "ofr_cash_by_size.json")
    cash_mnemonics = [
        "FPF-ALLQHF_GAVN10_CASHRATIO_AVERAGE",
        "FPF-ALLQHF_GAVN11TO50_CASHRATIO_AVERAGE",
        "FPF-ALLQHF_GAVN51_CASHRATIO_AVERAGE",
    ]
    cash_date, cash_percent = latest_values(cash_payload, cash_mnemonics)
    cash_values = [value / 100.0 for value in cash_percent]
    cash_low, cash_reference, cash_high = rounded_outward(
        min(cash_values), median(cash_values), max(cash_values), 2
    )

    financing_payload = load_json(RAW / "ofr_financing_maturity.json")
    financing_mnemonics = [
        "FPF-ALLQHF_FINANCINGLIQUIDTYLE1_PERCENT",
        "FPF-ALLQHF_FINANCINGLIQUIDTYGT1LE7_PERCENT",
        "FPF-ALLQHF_FINANCINGLIQUIDTYGT7LE90_PERCENT",
        "FPF-ALLQHF_FINANCINGLIQUIDTYGT90_PERCENT",
    ]
    financing_date, maturity_values = latest_values(
        financing_payload, financing_mnemonics
    )
    zero_one, two_seven, eight_ninety, over_ninety = maturity_values
    estimated_within_30 = zero_one + two_seven + eight_ninety * (23.0 / 83.0)

    market = fred_loss_summary(RAW / "fred_sp500.csv")
    anchors = {
        "anchor_version": "frl-v3-anchor-1.0",
        "generated_from_preserved_sources": True,
        "leverage": {
            "classification": "direct_data_anchor",
            "observation_date": leverage_date,
            "included_latest_values": leverage_values,
            "excluded_strategy_reason": "Derivative-heavy gross exposures are not comparable to a single long risky asset.",
            "low": leverage_low,
            "reference": leverage_reference,
            "high": leverage_high,
        },
        "cash_share_of_gross_assets": {
            "classification": "direct_data_anchor",
            "observation_date": cash_date,
            "included_latest_values": cash_values,
            "low": cash_low,
            "reference": cash_reference,
            "high": cash_high,
        },
        "financing_maturity": {
            "classification": "direct_data_anchor",
            "observation_date": financing_date,
            "percent_0_to_1_day": zero_one,
            "percent_2_to_7_days": two_seven,
            "percent_8_to_90_days": eight_ninety,
            "percent_over_90_days": over_ninety,
            "estimated_percent_within_30_days": estimated_within_30,
            "use": "Documents rollover exposure only; not a withdrawal probability.",
        },
        "market_loss_21_trading_days": {
            "classification": "direct_data_anchor",
            **market,
            "reference_quantile": "q95",
            "reference_loss": market["loss_q95"],
        },
        "funding_withdrawal_scenario": {
            "classification": "official_scenario_anchor",
            "fractions_of_initial_liabilities": [0.05, 0.10, 0.20],
        },
        "price_impact": {
            "classification": "official_scenario_anchor",
            "reference_sale_fraction": 0.10,
            "benchmark_lambda": 0.15,
            "high_lambda": 0.30,
            "benchmark_loss_at_reference_sale": 1.0 - math.exp(-0.15 * 0.10),
            "high_loss_at_reference_sale": 1.0 - math.exp(-0.30 * 0.10),
        },
        "normalized_parameters": {
            "market_event_probabilities": [0.05, 0.10, 0.20],
            "funding_noise_sigma": 0.20,
            "funding_multiplier_range": [0.75, 1.25],
        },
        "pilot_stress_budgets": [0.10, 0.15, 0.20, 0.25, 0.30],
    }
    output = ANCHORS / "parameter_anchors.json"
    output.write_text(
        json.dumps(anchors, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
