# --- START OF FILE environments/__init__.py ---

from .football import FootballEnv
from .running import RunningEnv
from .table_hockey import TableHockeyEnv
from .wrestling import WrestlingEnv
from .running_competition import RunningCompetitionEnv

# Mapping from environment name to the corresponding class
ENV_REGISTRY = {
    "football": FootballEnv,
    "running": RunningEnv,
    "table-hockey": TableHockeyEnv,
    "wrestling": WrestlingEnv,
    "running_competition": RunningCompetitionEnv,
}

def make(env_name, map_id=None, map_config=None):
    """
    Creates an environment instance from the given name.
    - For 'running-competition', it requires a `map_id`.
    - For other envs, it can optionally take a `map_config`.
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}.")
    
    env_class = ENV_REGISTRY[env_name]

    # --- Conditional Instantiation ---
    # Check which type of environment we are creating and provide
    # only the arguments it expects.
    
    if env_name == "running_competition":
        if map_id is None:
            # For the dummy env, we can default to 1
            print("Warning: map_id not provided for running-competition. Defaulting to 1.")
            map_id = 1
        return env_class(map_id=map_id)
    else:
        # For all other environments, call the constructor without map_id.
        # It will use its default internal map generation.
        # This assumes they have a __init__(self, map=None) signature.
        return env_class(map=map_config)