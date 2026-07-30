"""Generate EVA config TOML from a base template with key=value overrides.

CLI usage:
    python gen_config.py --base config.toml \\
        --set emitter.batch_size=500 --set run.n_generations=300 \\
        --output results/batch1/config.toml

Programmatic usage:
    from gen_config import generate_config, deep_set
    overrides = {"emitter.batch_size": 500, "run.n_generations": 300}
    generate_config("config.toml", overrides, "results/foo/config.toml")
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import toml

NestedDict = dict[str, "NestedDict | str | int | float | bool | list"]


def deep_set(cfg: NestedDict, key: str, value: str | int | float | bool | list) -> None:
    """Set a nested key like ``"emitter.batch_size"`` in a dict-of-dicts."""
    parts = key.split(".")
    d = cfg
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
        if not isinstance(d, dict):
            raise ValueError(f"Key '{key}' conflicts with non-dict value at '{part}'")

    # Type coercion
    typed: str | int | float | bool | list = value
    if isinstance(value, str):
        for cast in (int, float):
            try:
                typed = cast(value)
                break
            except (ValueError, TypeError):
                pass
        if value.lower() in ("true", "false"):
            typed = value.lower() == "true"

    d[parts[-1]] = typed


def generate_config(
    base_path: str | Path,
    overrides: dict[str, str | int | float | bool | list],
    output_path: str | Path,
) -> Path:
    """Read a base TOML, apply *overrides*, and write to *output_path*.

    Parameters
    ----------
    base_path : str or Path
        Path to the base ``config.toml`` template.
    overrides : dict
        Mapping of dotted keys to values, e.g.
        ``{"emitter.batch_size": 500, "run.n_generations": 300}``.
    output_path : str or Path
        Where to write the generated config.

    Returns
    -------
    Path
        The resolved output path.
    """
    base_path = Path(base_path)
    output_path = Path(output_path)

    with open(base_path) as f:
        cfg: NestedDict = toml.load(f)

    for key, value in overrides.items():
        deep_set(cfg, key, value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        toml.dump(cfg, f)

    return output_path


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EVA config TOML")
    parser.add_argument("--base", default="config.toml", help="Base config TOML template")
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        help="Key=value override, e.g. emitter.batch_size=500. Can be repeated.",
    )
    parser.add_argument("--output", required=True, help="Output TOML path")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_cli()
    overrides: dict[str, str | int | float | bool | list] = {}
    for kv in args.overrides:
        key, _, value = kv.partition("=")
        overrides[key] = value

    out = generate_config(args.base, overrides, args.output)
    print(f"Config written → {out}")


if __name__ == "__main__":
    main()
