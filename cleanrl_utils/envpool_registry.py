"""
EnvPool Environment ID Registry

Central source of truth for mapping legacy environment IDs (gymnasium-style)
to EnvPool-native IDs.

Mappings:
- Atari: NoFrameskip-v4 -> v5 (e.g., BreakoutNoFrameskip-v4 -> Breakout-v5)
- MuJoCo: v4 stays v4 (validated against EnvPool supported list)
- DMC: dm_control/<domain>-<task>-v0 -> DomainTask-v1 (e.g., dm_control/cheetah-run-v0 -> CheetahRun-v1)

Usage:
    from cleanrl_utils.envpool_registry import get_envpool_id, validate_envpool_id
    
    envpool_id = get_envpool_id("BreakoutNoFrameskip-v4")  # Returns "Breakout-v5"
    is_valid, backend = validate_envpool_id("CheetahRun-v1")  # Returns (True, "dm")
"""

from typing import Literal, Tuple, Optional
import re


# =============================================================================
# EnvPool-supported environment lists (from docs)
# =============================================================================

# MuJoCo gym environments supported by EnvPool 0.8.4
ENVPOOL_MUJOCO_IDS = {
    "Ant-v3", "Ant-v4",
    "HalfCheetah-v3", "HalfCheetah-v4",
    "Hopper-v3", "Hopper-v4",
    "Humanoid-v3", "Humanoid-v4",
    "HumanoidStandup-v2", "HumanoidStandup-v4",
    "InvertedDoublePendulum-v2", "InvertedDoublePendulum-v4",
    "InvertedPendulum-v2", "InvertedPendulum-v4",
    "Pusher-v2", "Pusher-v4",
    "Reacher-v2", "Reacher-v4",
    "Swimmer-v3", "Swimmer-v4",
    "Walker2d-v3", "Walker2d-v4",
}

# DMC environments supported by EnvPool 0.8.4
# Format: DomainTask-v1 (CamelCase)
ENVPOOL_DMC_IDS = {
    # Acrobot
    "AcrobotSwingup-v1", "AcrobotSwingupSparse-v1",
    # Ball in cup
    "BallInCupCatch-v1",
    # Cartpole
    "CartpoleBalance-v1", "CartpoleBalanceSparse-v1",
    "CartpoleSwingup-v1", "CartpoleSwingupSparse-v1",
    "CartpoleTwoPoles-v1", "CartpoleThreePoles-v1",
    # Cheetah
    "CheetahRun-v1",
    # Finger
    "FingerSpin-v1", "FingerTurnEasy-v1", "FingerTurnHard-v1",
    # Fish
    "FishSwim-v1", "FishUpright-v1",
    # Hopper
    "HopperStand-v1", "HopperHop-v1",
    # Manipulator
    "ManipulatorBringBall-v1", "ManipulatorBringPeg-v1",
    "ManipulatorInsertBall-v1", "ManipulatorInsertPeg-v1",
    # Humanoid
    "HumanoidStand-v1", "HumanoidWalk-v1", "HumanoidRun-v1", "HumanoidRunPureState-v1",
    # Humanoid CMU
    "HumanoidCMUStand-v1", "HumanoidCMURun-v1",
    # Pendulum
    "PendulumSwingup-v1",
    # Point mass
    "PointMassEasy-v1", "PointMassHard-v1",
    # Reacher
    "ReacherEasy-v1", "ReacherHard-v1",
    # Swimmer
    "SwimmerSwimmer6-v1", "SwimmerSwimmer15-v1",
    # Walker
    "WalkerRun-v1", "WalkerStand-v1", "WalkerWalk-v1",
}

# Atari environments supported by EnvPool 0.8.4
# Format: GameName-v5 (CamelCase)
ENVPOOL_ATARI_IDS = {
    "Adventure-v5", "AirRaid-v5", "Alien-v5", "Amidar-v5", "Assault-v5",
    "Asterix-v5", "Asteroids-v5", "Atlantis-v5", "Atlantis2-v5", "Backgammon-v5",
    "BankHeist-v5", "BasicMath-v5", "BattleZone-v5", "BeamRider-v5", "Berzerk-v5",
    "Blackjack-v5", "Bowling-v5", "Boxing-v5", "Breakout-v5", "Carnival-v5",
    "Casino-v5", "Centipede-v5", "ChopperCommand-v5", "CrazyClimber-v5", "Crossbow-v5",
    "Darkchambers-v5", "Defender-v5", "DemonAttack-v5", "DonkeyKong-v5", "DoubleDunk-v5",
    "Earthworld-v5", "ElevatorAction-v5", "Enduro-v5", "Entombed-v5", "Et-v5",
    "FishingDerby-v5", "FlagCapture-v5", "Freeway-v5", "Frogger-v5", "Frostbite-v5",
    "Galaxian-v5", "Gopher-v5", "Gravitar-v5", "Hangman-v5", "HauntedHouse-v5",
    "Hero-v5", "HumanCannonball-v5", "IceHockey-v5", "Jamesbond-v5", "JourneyEscape-v5",
    "Kaboom-v5", "Kangaroo-v5", "KeystoneKapers-v5", "KingKong-v5", "Klax-v5",
    "Koolaid-v5", "Krull-v5", "KungFuMaster-v5", "LaserGates-v5", "LostLuggage-v5",
    "MarioBros-v5", "MiniatureGolf-v5", "MontezumaRevenge-v5", "MrDo-v5", "MsPacman-v5",
    "NameThisGame-v5", "Othello-v5", "Pacman-v5", "Phoenix-v5", "Pitfall-v5",
    "Pitfall2-v5", "Pong-v5", "Pooyan-v5", "PrivateEye-v5", "Qbert-v5",
    "Riverraid-v5", "RoadRunner-v5", "Robotank-v5", "Seaquest-v5", "SirLancelot-v5",
    "Skiing-v5", "Solaris-v5", "SpaceInvaders-v5", "SpaceWar-v5", "StarGunner-v5",
    "Superman-v5", "Surround-v5", "Tennis-v5", "Tetris-v5", "TicTacToe3d-v5",
    "TimePilot-v5", "Trondead-v5", "Turmoil-v5", "Tutankham-v5", "UpNDown-v5",
    "Venture-v5", "VideoCheckers-v5", "VideoChess-v5", "VideoCube-v5", "VideoPinball-v5",
    "WizardOfWor-v5", "WordZapper-v5", "YarsRevenge-v5", "Zaxxon-v5",
}


# =============================================================================
# Legacy ID -> EnvPool ID mappings
# =============================================================================

def _atari_legacy_to_envpool(legacy_id: str) -> Optional[str]:
    """
    Convert legacy Atari ID (NoFrameskip-v4) to EnvPool ID (v5).
    
    Examples:
        BreakoutNoFrameskip-v4 -> Breakout-v5
        PongNoFrameskip-v4 -> Pong-v5
        SpaceInvadersNoFrameskip-v4 -> SpaceInvaders-v5
    """
    # Pattern: GameNameNoFrameskip-v4
    match = re.match(r"^(.+)NoFrameskip-v4$", legacy_id)
    if match:
        game_name = match.group(1)
        envpool_id = f"{game_name}-v5"
        if envpool_id in ENVPOOL_ATARI_IDS:
            return envpool_id
    return None


def _dmc_legacy_to_envpool(legacy_id: str) -> Optional[str]:
    """
    Convert legacy DMC ID (dm_control/<domain>-<task>-v0) to EnvPool ID (DomainTask-v1).
    
    Examples:
        dm_control/cheetah-run-v0 -> CheetahRun-v1
        dm_control/walker-walk-v0 -> WalkerWalk-v1
        dm_control/humanoid-walk-v0 -> HumanoidWalk-v1
        dm_control/ball_in_cup-catch-v0 -> BallInCupCatch-v1
    """
    # Pattern: dm_control/<domain>-<task>-v0
    match = re.match(r"^dm_control/(.+)-(.+)-v0$", legacy_id)
    if match:
        domain = match.group(1)
        task = match.group(2)
        
        # Convert to CamelCase: domain_name -> DomainName
        def to_camel_case(s: str) -> str:
            parts = s.replace("_", "-").split("-")
            return "".join(p.capitalize() for p in parts)
        
        domain_camel = to_camel_case(domain)
        task_camel = to_camel_case(task)
        envpool_id = f"{domain_camel}{task_camel}-v1"
        
        if envpool_id in ENVPOOL_DMC_IDS:
            return envpool_id
    return None


def _mujoco_legacy_to_envpool(legacy_id: str) -> Optional[str]:
    """
    Validate MuJoCo ID (v4) against EnvPool supported list.
    MuJoCo IDs are the same in gymnasium and EnvPool.
    
    Examples:
        HalfCheetah-v4 -> HalfCheetah-v4 (validated)
        Hopper-v4 -> Hopper-v4 (validated)
    """
    if legacy_id in ENVPOOL_MUJOCO_IDS:
        return legacy_id
    return None


# =============================================================================
# Public API
# =============================================================================

EnvType = Literal["mujoco", "dmc", "atari", "unknown"]
Backend = Literal["gym", "dm"]


def detect_env_type(env_id: str) -> EnvType:
    """
    Detect the environment type from the ID.
    
    Returns:
        "atari" for Atari games
        "dmc" for DeepMind Control Suite
        "mujoco" for MuJoCo gym
        "unknown" if cannot determine
    """
    # Check if it's a legacy DMC ID
    if env_id.startswith("dm_control/"):
        return "dmc"
    
    # Check if it's a legacy Atari ID
    if "NoFrameskip-v4" in env_id:
        return "atari"
    
    # Check if it's an EnvPool Atari ID
    if env_id in ENVPOOL_ATARI_IDS:
        return "atari"
    
    # Check if it's an EnvPool DMC ID
    if env_id in ENVPOOL_DMC_IDS:
        return "dmc"
    
    # Check if it's an EnvPool/gym MuJoCo ID
    if env_id in ENVPOOL_MUJOCO_IDS:
        return "mujoco"
    
    return "unknown"


def get_envpool_id(legacy_id: str) -> str:
    """
    Convert a legacy environment ID to its EnvPool-native equivalent.
    
    Args:
        legacy_id: The legacy environment ID (gymnasium-style)
        
    Returns:
        The EnvPool-native ID
        
    Raises:
        ValueError: If the ID cannot be mapped to an EnvPool ID
        
    Examples:
        >>> get_envpool_id("BreakoutNoFrameskip-v4")
        "Breakout-v5"
        >>> get_envpool_id("dm_control/cheetah-run-v0")
        "CheetahRun-v1"
        >>> get_envpool_id("HalfCheetah-v4")
        "HalfCheetah-v4"
    """
    # Already an EnvPool ID?
    if legacy_id in ENVPOOL_MUJOCO_IDS | ENVPOOL_DMC_IDS | ENVPOOL_ATARI_IDS:
        return legacy_id
    
    # Try Atari conversion
    envpool_id = _atari_legacy_to_envpool(legacy_id)
    if envpool_id:
        return envpool_id
    
    # Try DMC conversion
    envpool_id = _dmc_legacy_to_envpool(legacy_id)
    if envpool_id:
        return envpool_id
    
    # Try MuJoCo validation
    envpool_id = _mujoco_legacy_to_envpool(legacy_id)
    if envpool_id:
        return envpool_id
    
    raise ValueError(
        f"Cannot map '{legacy_id}' to an EnvPool ID. "
        f"Supported types: Atari (*NoFrameskip-v4), DMC (dm_control/*-v0), MuJoCo (*-v4)"
    )


def get_envpool_backend(env_id: str) -> Backend:
    """
    Get the EnvPool backend to use for an environment.
    
    Args:
        env_id: The EnvPool environment ID
        
    Returns:
        "gym" for MuJoCo and Atari (use envpool.make)
        "dm" for DMC (use envpool.make_dm)
    """
    env_type = detect_env_type(env_id)
    if env_type == "dmc":
        return "dm"
    return "gym"


def validate_envpool_id(env_id: str) -> Tuple[bool, Optional[Backend]]:
    """
    Validate that an environment ID is supported by EnvPool.
    
    Args:
        env_id: The environment ID to validate
        
    Returns:
        Tuple of (is_valid, backend) where backend is "gym" or "dm" if valid, None otherwise
    """
    # Check direct EnvPool IDs
    if env_id in ENVPOOL_MUJOCO_IDS:
        return (True, "gym")
    if env_id in ENVPOOL_ATARI_IDS:
        return (True, "gym")
    if env_id in ENVPOOL_DMC_IDS:
        return (True, "dm")
    
    # Try to convert from legacy format
    try:
        envpool_id = get_envpool_id(env_id)
        backend = get_envpool_backend(envpool_id)
        return (True, backend)
    except ValueError:
        return (False, None)


def make_envpool_env(
    env_id: str,
    num_envs: int = 1,
    seed: int = 1,
    **kwargs
):
    """
    Create an EnvPool environment from any supported ID format.
    
    Args:
        env_id: Legacy or EnvPool environment ID
        num_envs: Number of parallel environments
        seed: Random seed
        **kwargs: Additional arguments passed to envpool.make/make_dm
        
    Returns:
        EnvPool environment
    """
    import envpool
    
    envpool_id = get_envpool_id(env_id)
    backend = get_envpool_backend(envpool_id)
    
    if backend == "dm":
        return envpool.make_dm(envpool_id, num_envs=num_envs, seed=seed, **kwargs)
    else:
        return envpool.make(envpool_id, env_type="gym", num_envs=num_envs, seed=seed, **kwargs)


# =============================================================================
# Benchmark environment lists
# =============================================================================

# Standard benchmark environments used in ICML experiments

BENCHMARK_MUJOCO_IDS = [
    "HalfCheetah-v4",
    "Hopper-v4", 
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
    "Swimmer-v4",
    "InvertedDoublePendulum-v4",
]

BENCHMARK_DMC_LEGACY_IDS = [
    "dm_control/cheetah-run-v0",
    "dm_control/walker-walk-v0",
    "dm_control/walker-run-v0",
    "dm_control/hopper-stand-v0",
    "dm_control/hopper-hop-v0",
    "dm_control/humanoid-walk-v0",
    "dm_control/humanoid-run-v0",
    "dm_control/fish-swim-v0",
    "dm_control/acrobot-swingup-v0",
    "dm_control/cartpole-swingup-v0",
]

BENCHMARK_DMC_ENVPOOL_IDS = [
    "CheetahRun-v1",
    "WalkerWalk-v1",
    "WalkerRun-v1",
    "HopperStand-v1",
    "HopperHop-v1",
    "HumanoidWalk-v1",
    "HumanoidRun-v1",
    "FishSwim-v1",
    "AcrobotSwingup-v1",
    "CartpoleSwingup-v1",
]

BENCHMARK_ATARI_LEGACY_IDS = [
    "BreakoutNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
]

BENCHMARK_ATARI_ENVPOOL_IDS = [
    "Breakout-v5",
    "Pong-v5",
    "SpaceInvaders-v5",
    "Seaquest-v5",
    "BeamRider-v5",
    "Enduro-v5",
    "Qbert-v5",
    "MsPacman-v5",
    "Asterix-v5",
    "RoadRunner-v5",
]


def get_benchmark_envpool_ids(
    include_mujoco: bool = True,
    include_dmc: bool = True,
    include_atari: bool = True,
) -> dict:
    """
    Get all benchmark environment IDs in EnvPool format.
    
    Returns:
        Dict with keys "mujoco", "dmc", "atari" containing lists of EnvPool IDs
    """
    result = {}
    if include_mujoco:
        result["mujoco"] = BENCHMARK_MUJOCO_IDS.copy()
    if include_dmc:
        result["dmc"] = BENCHMARK_DMC_ENVPOOL_IDS.copy()
    if include_atari:
        result["atari"] = BENCHMARK_ATARI_ENVPOOL_IDS.copy()
    return result


def get_legacy_to_envpool_mapping() -> dict:
    """
    Get a complete mapping from legacy IDs to EnvPool IDs for benchmark environments.
    
    Returns:
        Dict mapping legacy_id -> envpool_id
    """
    mapping = {}
    
    # MuJoCo (same IDs)
    for env_id in BENCHMARK_MUJOCO_IDS:
        mapping[env_id] = env_id
    
    # DMC
    for legacy_id, envpool_id in zip(BENCHMARK_DMC_LEGACY_IDS, BENCHMARK_DMC_ENVPOOL_IDS):
        mapping[legacy_id] = envpool_id
    
    # Atari
    for legacy_id, envpool_id in zip(BENCHMARK_ATARI_LEGACY_IDS, BENCHMARK_ATARI_ENVPOOL_IDS):
        mapping[legacy_id] = envpool_id
    
    return mapping


# =============================================================================
# Self-check functions for testing
# =============================================================================

def check_all_benchmark_ids_valid() -> bool:
    """
    Verify that all benchmark environment IDs can be created with EnvPool.
    
    Returns:
        True if all IDs are valid, False otherwise
    """
    all_valid = True
    errors = []
    
    benchmark_ids = get_benchmark_envpool_ids()
    
    for env_type, env_ids in benchmark_ids.items():
        for env_id in env_ids:
            is_valid, backend = validate_envpool_id(env_id)
            if not is_valid:
                all_valid = False
                errors.append(f"{env_type}/{env_id}: not valid")
    
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
    
    return all_valid


def check_legacy_mappings_valid() -> bool:
    """
    Verify that all legacy ID mappings produce valid EnvPool IDs.
    
    Returns:
        True if all mappings are valid, False otherwise
    """
    mapping = get_legacy_to_envpool_mapping()
    all_valid = True
    errors = []
    
    for legacy_id, expected_envpool_id in mapping.items():
        try:
            actual_envpool_id = get_envpool_id(legacy_id)
            if actual_envpool_id != expected_envpool_id:
                all_valid = False
                errors.append(f"{legacy_id}: expected {expected_envpool_id}, got {actual_envpool_id}")
        except ValueError as e:
            all_valid = False
            errors.append(f"{legacy_id}: {e}")
    
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
    
    return all_valid


if __name__ == "__main__":
    # Self-test when run directly
    print("Checking benchmark IDs...")
    if check_all_benchmark_ids_valid():
        print("All benchmark IDs are valid.")
    else:
        print("Some benchmark IDs are invalid!")
    
    print("\nChecking legacy ID mappings...")
    if check_legacy_mappings_valid():
        print("All legacy ID mappings are valid.")
    else:
        print("Some legacy ID mappings are invalid!")
    
    print("\nExample conversions:")
    examples = [
        "BreakoutNoFrameskip-v4",
        "dm_control/cheetah-run-v0",
        "HalfCheetah-v4",
        "dm_control/humanoid-walk-v0",
        "SpaceInvadersNoFrameskip-v4",
    ]
    for legacy_id in examples:
        try:
            envpool_id = get_envpool_id(legacy_id)
            backend = get_envpool_backend(envpool_id)
            print(f"  {legacy_id} -> {envpool_id} (backend: {backend})")
        except ValueError as e:
            print(f"  {legacy_id} -> ERROR: {e}")
