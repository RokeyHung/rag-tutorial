from __future__ import annotations

import os


TRUE_SET = {"1", "true", "yes", "y", "on"}
FALSE_SET = {"0", "false", "no", "n", "off"}


def env_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw != "" else default


def env_bool(name: str, default: bool = False) -> bool:
    raw = env_str(name, None)
    if raw is None:
        return default
    v = raw.lower()
    if v in TRUE_SET:
        return True
    if v in FALSE_SET:
        return False
    return default


def env_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw = env_str(name, None)
    if raw is None:
        v = default
    else:
        try:
            v = int(raw)
        except Exception:
            v = default

    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v

