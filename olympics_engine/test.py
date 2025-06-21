import os
import sys
from pathlib import Path
import torch
import argparse

# ─── 0) Compute root_dir so that "olympics_engine" and project root are on Python path ───
# test.py lives in:
#   /Users/kuzeyarar/Desktop/IJCAI_2023/IJCAI2023/olympics_engine/test.py
# We need both "olympics_engine" and the folder containing train2avoid_ppo.py to be importable.

olympics_dir = Path(__file__).resolve().parent
project_root = str(olympics_dir.parent)  # "/Users/kuzeyarar/Desktop/IJCAI_2023/IJCAI2023"
sys.path.append(project_root)            # allows "import train2avoid_ppo"
sys.path.append(os.path.join(project_root, "train"))  # allows "from train.algo.ppo import PPO"
sys.path.append(project_root)            # also makes olympics_engine importable

print(sys.path)

from olympics_engine.agent import *
from olympics_engine.generator import create_scenario
import time
import random
import json
import imageio
import pygame  # needed for frame capture

from scenario import Running, table_hockey, football, wrestling, billiard, \
    curling, billiard_joint, curling_long, curling_competition, Running_competition, billiard_competition, Seeks
from AI_olympics import AI_Olympics

# PPO and its networks
from train.algo.ppo import PPO

# Import the Learn2Avoid environment class defined inside train2avoid_ppo.py
from train2avoid_ppo import env_test as Learn2AvoidEnv


def store(record, name):
    with open('logs/' + name + '.json', 'w') as f:
        f.write(json.dumps(record))


def load_record(path):
    with open(path, "rb") as file:
        return json.load(file)


RENDER = True  # Set to True if you want to render the game

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--map', default=None, type=str,
        help='Which subgame to run (e.g. "seeks", "football", etc.). If omitted, uses --game_name.'
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--load_model",default=True, action='store_true',
                        help="Whether to load a pretrained PPO model")
    parser.add_argument("--game_name", default="Learn2Avoid", type=str,
                        help="Name of the folder under train/models (e.g. Learn2Avoid). Also used as environment if --map is not provided.")
    parser.add_argument("--load_run", default=1, type=int,
                        help="Which run directory to load (e.g. run1)")
    parser.add_argument("--load_episode", default=10, type=int,
                        help="Which episode checkpoint to load (i.e. actor_<episode>.pth)")
    parser.add_argument("--capture_frames", default=False, action='store_true',
                        help="If set, capture frames for GIF output")
    parser.add_argument("--render", action='store_true', default=False,
                        help='Render the game visually. If not set, the game runs without rendering.')
    args = parser.parse_args()
    RENDER = args.render

    # If --map is not provided, default to the training environment name
    if args.map is None:
        args.map = args.game_name

    random.seed(args.seed)

    # ──── ACTION‐SPACE MAPPING ─────────────────────────────────────────────────────
    actions_map = {
        0: [-100, -30],  1: [-100, -18],  2: [-100, -6],   3: [-100, 6],   4: [-100, 18],  5: [-100, 30],
        6: [-40, -30],   7: [-40, -18],   8: [-40, -6],    9: [-40, 6],    10: [-40, 18],  11: [-40, 30],
        12: [20, -30],   13: [20, -18],   14: [20, -6],    15: [20, 6],    16: [20, 18],   17: [20, 30],
        18: [80, -30],   19: [80, -18],   20: [80, -6],    21: [80, 6],    22: [80, 18],   23: [80, 30],
        24: [140, -30],  25: [140, -18],  26: [140, -6],   27: [140, 6],   28: [140, 18],  29: [140, 30],
        30: [200, -30],  31: [200, -18],  32: [200, -6],   33: [200, 6],   34: [200, 18],  35: [200, 30]
    }

    # ─── Select and instantiate the environment based on args.map ────────────────
    if args.map == "Learn2Avoid":
        # Use the env_test class from train2avoid_ppo.py as the Learn2Avoid environment
        game = Learn2AvoidEnv()
        agent_num = 1
    else:
        # For other subgames (e.g. "seeks", "football"), use create_scenario
        Gamemap = create_scenario(args.map)
        if args.map == 'running':
            game = Running(Gamemap); agent_num = 2
        elif args.map == 'running-competition':
            map_id = random.randint(1, 10)
            game = Running_competition(meta_map=Gamemap, map_id=map_id); agent_num = 2
        elif args.map == 'seeks':
            game = Seeks(Gamemap); agent_num = 2
        elif args.map == 'table-hockey':
            game = table_hockey(Gamemap); agent_num = 2
        elif args.map == 'football':
            game = football(Gamemap); agent_num = 2
        elif args.map == 'wrestling':
            game = wrestling(Gamemap); agent_num = 2
        elif args.map == 'billiard':
            game = billiard(Gamemap); agent_num = 2
        elif args.map == 'billiard-competition':
            game = billiard_competition(Gamemap); agent_num = 2
        elif args.map == 'curling':
            game = curling(Gamemap); agent_num = 2
        elif args.map == 'curling-joint':
            game = curling_joint(Gamemap); agent_num = 2
        elif args.map == 'billiard-joint':
            game = billiard_joint(Gamemap); agent_num = 2
        elif args.map == 'curling-long':
            game = curling_long(Gamemap); agent_num = 2
        elif args.map == 'curling-competition':
            game = curling_competition(Gamemap); agent_num = 2
        elif args.map == 'all':
            game = AI_Olympics(random_selection=False, minimap=False); agent_num = 2
        elif args.map == 'all_v2':
            game = AI_Olympics(random_selection=False, minimap=False, vis=300, vis_clear=5); agent_num = 2
        else:
            raise ValueError(f"Unknown map/environment: {args.map}")

    # ─── Instantiate Agents ─────────────────────────────────────────────────────────
    if args.load_model:
        model = PPO()

        # Build run_dir pointing to train/models/<game_name>/run<load_run>
        run_dir = os.path.join(
            project_root,#/Users/kuzeyarar/Desktop/IJCAI_2023/IJCAI2023/
            "rl_trainer",
            "models",
            args.game_name,
            f"run{args.load_run}"
        )
        print("Begin to load model:")
        print("  run_dir:", run_dir)

        # Construct paths to actor_<episode>.pth and critic_<episode>.pth
        actor_path = os.path.join(
            run_dir,
            "trained_model",
            f"actor_{args.load_episode}.pth"
        )
        critic_path = os.path.join(
            run_dir,
            "trained_model",
            f"critic_{args.load_episode}.pth"
        )
        print("  Actor path: ", actor_path)
        print("  Critic path:", critic_path)

        # Verify that the checkpoint files exist
        if not os.path.isfile(actor_path) or not os.path.isfile(critic_path):
            print("\n[ERROR] Checkpoint file(s) not found!  Check the paths above.\n")
            sys.exit(1)

        # Load the PPO weights (the PPO.load method expects run_dir and episode)
        try:
            model.load(run_dir, episode=args.load_episode)
            print("[INFO] ✓ PPO successfully loaded from run_dir")
        except Exception as e:
            print(f"[ERROR] Failed to load PPO from {run_dir}:")
            print("         ", str(e))
            sys.exit(1)

        agent = model
        # Opponent remains random (or load a second PPO if you trained two agents)
        rand_agent = random_agent()
    else:
        agent = random_agent()
        rand_agent = random_agent()

    # ─── Run the episode ───────────────────────────────────────────────────────────
    obs = game.reset()
   
    frames = []
    done = False
    step = 0
    if RENDER:
        game.render()

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    time_epi_s = time.time()
    while not done:
        step += 1

        if agent_num == 2:
            if args.load_model:
                raw = obs[0]['agent_obs'] if isinstance(obs[0], dict) else obs[0]
                obs0_flat = raw.flatten()
                with torch.no_grad():
                    action_index, _ = agent.select_action(obs0_flat, False)
                action1 = actions_map[action_index]
            else:
                raw = obs[0]['agent_obs'] if isinstance(obs[0], dict) else obs[0]
                action1 = agent.act(raw)

            action2 = rand_agent.act(obs[1])
            action = [action1, action2]

        else:  # agent_num == 1 (Learn2Avoid)
            if args.load_model:
                raw = obs[0]['agent_obs'] if isinstance(obs[0], dict) else obs[0]
                obs0_flat = raw.flatten()
                
                with torch.no_grad():
                    action_index, _ = agent.select_action(obs0_flat, False)
                action1 = actions_map[action_index]
            else:
                raw = obs[0]['agent_obs'] if isinstance(obs[0], dict) else obs[0]
                action1 = agent.act(raw)

            action = [action1]

        obs, reward, done, _ = game.step(action)
        print(f"reward = {reward}")

        if RENDER:
            game.render()
            if args.capture_frames:
                screen = pygame.display.get_surface()
                if screen is not None:
                    img = pygame.surfarray.array3d(screen)
                    img = img.swapaxes(0, 1)
                    frames.append(img)
                else:
                    print(f"Warning: Screen surface not available at step {step}")

    duration_t = time.time() - time_epi_s
    print(
        "episode duration:", duration_t,
        "step:", step,
        "time-per-step:", (duration_t) / step
    )

    # Save gameplay as a GIF if any frames were captured
    save_folder = '/Users/kuzeyarar/Desktop/Gameplay'
    os.makedirs(save_folder, exist_ok=True)
    gif_filename = f"{args.map}.gif"
    save_path = os.path.join(save_folder, gif_filename)

    if args.capture_frames and len(frames) > 0:
        imageio.mimsave(save_path, frames, fps=30)
        print("Saved successfully!")
    elif args.capture_frames:
        print("No frames captured. GIF not saved.")

# Example to run:
# python test.py --map Learn2Avoid --seed 1 --load_model --game_name Learn2Avoid --load_run 1 --load_episode 10
# python test.py --map table-hockey --seed 1 --load_model --game_name table-hockey --load_run 1 --load_episode 33000
# python test.py --map Learn2Avoid --seed 1  --game_name Learn2Avoid --load_run 8 --load_episode 700