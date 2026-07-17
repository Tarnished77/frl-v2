"""Order-independent deterministic random streams."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .config import ModelConfig


STREAM_NAMES = (
    "initial_leverage",
    "initial_cash_share",
    "payment_multiplier",
    "liquidity_noise",
    "market_events",
)


def seed_from_parts(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def generator_for(namespace: str, replication_id: str, stream_name: str) -> np.random.Generator:
    if stream_name not in STREAM_NAMES:
        raise ValueError(f"Unknown random stream: {stream_name}")
    return np.random.default_rng(seed_from_parts(namespace, replication_id, stream_name))


@dataclass(frozen=True)
class RandomInputs:
    leverage_u: np.ndarray
    cash_share_u: np.ndarray
    payment_multiplier_u: np.ndarray
    liquidity_z: np.ndarray
    market_event_u: np.ndarray


def make_random_inputs(config: ModelConfig, namespace: str, replication_id: str) -> RandomInputs:
    population = config.population
    horizon = config.horizon
    return RandomInputs(
        leverage_u=generator_for(namespace, replication_id, "initial_leverage").random(population),
        cash_share_u=generator_for(namespace, replication_id, "initial_cash_share").random(population),
        payment_multiplier_u=generator_for(namespace, replication_id, "payment_multiplier").random(population),
        liquidity_z=generator_for(namespace, replication_id, "liquidity_noise").normal(
            size=(horizon, population)
        ),
        market_event_u=generator_for(namespace, replication_id, "market_events").random(horizon),
    )


def stream_seed_manifest(namespace: str, replication_ids: list[str]) -> dict[str, dict[str, str]]:
    return {
        replication_id: {
            stream_name: str(seed_from_parts(namespace, replication_id, stream_name))
            for stream_name in STREAM_NAMES
        }
        for replication_id in replication_ids
    }
