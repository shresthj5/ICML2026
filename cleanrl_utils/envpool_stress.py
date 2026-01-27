"""
Helpers for applying stress presets in EnvPool-native scripts.

The original (gymnasium) scripts use wrapper-based stress (`stress_wrappers.py`).
EnvPool envs cannot be wrapped the same way, so EnvPool scripts implement
array-level transforms and (optionally) hyperparameter overrides.

This module keeps the **CLI interface** compatible by:
- Accepting `--stress_preset`
- Applying any preset `cli_args` as in the original stress suite
- Mapping known wrapper presets onto array-level knobs:
  - ObsNoiseWrapper -> obs_noise_std (and optional relative)
  - ObsDropoutWrapper -> obs_dropout_p
  - ActionDelayWrapper -> action_delay
  - RewardScaleWrapper -> reward_scale
"""

from __future__ import annotations

from typing import Any


def apply_stress_preset_to_args(args: Any) -> None:
    """
    Mutate `args` in-place by applying the preset.

    - Hyperparameter presets use `get_stress_preset_cli_args`
    - Env stress presets map wrapper definitions to scalar args, if present
    """
    preset_name = getattr(args, "stress_preset", "") or ""
    if not preset_name:
        return

    from cleanrl_utils.stress_wrappers import STRESS_PRESETS, get_stress_preset_cli_args

    # 1) Apply CLI hyperparameter overrides (e.g., vf_coef/clip/update_epochs/q_lr/etc.)
    cli_overrides = get_stress_preset_cli_args(preset_name)
    for k, v in cli_overrides.items():
        if hasattr(args, k):
            current = getattr(args, k)
            try:
                casted = type(current)(v)
            except Exception:
                casted = v
            setattr(args, k, casted)

    # 2) Map wrapper-based env stress to array-level knobs
    preset = STRESS_PRESETS.get(preset_name, {})
    wrappers = preset.get("wrappers", []) or []
    for w in wrappers:
        w_type = w.get("type", "")
        if w_type == "ObsNoiseWrapper":
            std = float(w.get("std", 0.0))
            if hasattr(args, "obs_noise_std"):
                args.obs_noise_std = max(float(getattr(args, "obs_noise_std")), std)
            # Optional: some scripts may support `obs_noise_relative`
            if hasattr(args, "obs_noise_relative") and "relative" in w:
                setattr(args, "obs_noise_relative", bool(w.get("relative", True)))
        elif w_type == "ObsDropoutWrapper":
            p = float(w.get("p", 0.0))
            if hasattr(args, "obs_dropout_p"):
                args.obs_dropout_p = max(float(getattr(args, "obs_dropout_p")), p)
        elif w_type == "ActionDelayWrapper":
            d = int(w.get("delay_steps", 0))
            if hasattr(args, "action_delay"):
                args.action_delay = max(int(getattr(args, "action_delay")), d)
        elif w_type == "RewardScaleWrapper":
            s = float(w.get("scale", 1.0))
            if hasattr(args, "reward_scale"):
                args.reward_scale = float(getattr(args, "reward_scale")) * s

