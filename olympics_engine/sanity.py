# File: /Users/kuzeyarar/Desktop/IJCAI_2023/IJCAI2023/olympics_engine/sanity.py

import os
import sys
from pathlib import Path
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity‐check PPO checkpoint paths")
    parser.add_argument(
        "--game_name", type=str, default="Learn2Avoid",
        help="Name of the game folder under train/models (e.g. Learn2Avoid)"
    )
    parser.add_argument(
        "--load_run", type=int, default=1,
        help="Which run directory to load (e.g. run1)"
    )
    parser.add_argument(
        "--episode", type=int, default=10,
        help="Which episode checkpoint to load (i.e. actor_<episode>.pth)"
    )
    args = parser.parse_args()

    # ─── 1) Determine base_dir ───────────────────────────────────────────────────
    # Since sanity.py is at:
    #   /…/IJCAI2023/olympics_engine/sanity.py
    # its parent folder (base_dir) is:
    base_dir = str(Path(__file__).resolve().parent)
    # base_dir == "/Users/kuzeyarar/Desktop/IJCAI_2023/IJCAI2023/olympics_engine"
    print("base_dir:", base_dir)

    # ─── 2) Build run_dir WITHOUT duplicating "olympics_engine" ─────────────────
    # The checkpoints live at:
    #   base_dir/train/models/<game_name>/run<load_run>/trained_model/actor_<episode>.pth
    run_dir = os.path.join(
        base_dir,
        "train",
        "models",
        args.game_name,         # e.g. "Learn2Avoid"
        f"run{args.load_run}"   # e.g. "run1"
    )
    print("run_dir:", run_dir)
    # ▶ Expected: "/…/IJCAI2023/olympics_engine/train/models/Learn2Avoid/run1"

    # ─── 3) Build actor_path and critic_path ────────────────────────────────────
    actor_path  = os.path.join(run_dir, "trained_model", f"actor_{args.episode}.pth")
    critic_path = os.path.join(run_dir, "trained_model", f"critic_{args.episode}.pth")

    print("Actor path:  ", actor_path)
    print("Critic path: ", critic_path)

    # ─── 4) Check existence ─────────────────────────────────────────────────────
    actor_exists = os.path.isfile(actor_path)
    critic_exists = os.path.isfile(critic_path)

    print("Actor exists?  ", actor_exists)
    print("Critic exists? ", critic_exists)

    if not (actor_exists and critic_exists):
        print("\nModel not found!  Please verify those paths above.")
        sys.exit(1)

    # ─── 5) (Optional) Load into your networks ─────────────────────────────────
    # Replace `YourActorNetwork` / `YourCriticNetwork` with your actual classes.
    # For example:
    # policy_net = YourActorNetwork(...)
    # value_net  = YourCriticNetwork(...)
    #
    # policy_net.load_state_dict(torch.load(actor_path))
    # value_net.load_state_dict(torch.load(critic_path))
    #
    # print("✓ Loaded successfully!")
    #
    # (Skip the above if you only want to sanity‐check the file locations.)
