"""
wandb_utils.py

Helper utilities to make Weights & Biases logging optional and safe-by-default.

Usage pattern
-------------
from wandb_utils import maybe_init_wandb, maybe_log, maybe_finish

run = maybe_init_wandb(...)
maybe_log(run, {...})
maybe_finish(run)

Security note
-------------
Do NOT hardcode API keys in code. W&B expects credentials via:
- environment variable WANDB_API_KEY or wandb_api_key
- `wandb login` (stored locally)

If no API key is set, we run in offline/disabled mode.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import os


def get_wandb_api_key() -> str:
    """
    Get W&B API key from environment variables.
    Checks both WANDB_API_KEY and wandb_api_key (case-insensitive support).
    """
    # Check standard WANDB_API_KEY first
    api_key = os.getenv("WANDB_API_KEY", "")
    if api_key:
        return api_key

    # Check lowercase variant (wandb_api_key)
    api_key = os.getenv("wandb_api_key", "")
    if api_key:
        # Set it to the standard variable so wandb can find it
        os.environ["WANDB_API_KEY"] = api_key
        return api_key

    return ""


def maybe_init_wandb(
    project: str,
    name: str,
    group: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    enable: bool = False,
) -> Optional[Any]:
    """
    Returns a wandb.Run-like object or None (if disabled).

    If enable=True but no API key is found, we fall back to offline mode
    so code still runs and produces local logs.

    Supports environment variables: WANDB_API_KEY, wandb_api_key
    """
    if not enable:
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        print("[wandb] not installed. Install with: pip install wandb")
        return None

    # Get API key from environment (supports both cases)
    api_key = get_wandb_api_key()

    if api_key:
        print(f"[wandb] API key found, logging to cloud.")
    else:
        print("[wandb] No API key found, running in offline mode.")
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        project=project,
        name=name,
        group=group,
        config=config or {},
        tags=tags or [],
    )
    return run


def maybe_log(run: Optional[Any], data: Dict[str, Any], step: Optional[int] = None) -> None:
    if run is None:
        return
    try:
        import wandb  # type: ignore

        wandb.log(data, step=step)
    except Exception:
        # Do not crash training if logging fails
        pass


def maybe_finish(run: Optional[Any]) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
