import os
import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
print(sys.path)
from olympics_engine.generator import create_scenario
import argparse
from olympics_engine.agent import *
import time

from scenario import Running, table_hockey, football, wrestling, billiard, \
    curling, billiard_joint, curling_long, curling_competition, Running_competition, billiard_competition, Seeks

from AI_olympics import AI_Olympics

from train.algo.ppo import PPO


import random
import json
import imageio
import pygame  # important for manual frame capture

def store(record, name):
    with open('logs/' + name + '.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = False  # Set to True if you want to render the game

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default="seeks", type=str,
                        help='running/table-hockey/football/wrestling/billiard/curling/all/all_v2')
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    
    args.map = 'seeks'

    for i in range(1):
        if 'all' not in args.map:
            Gamemap = create_scenario(args.map)
        
        if args.map == 'running':
            game = Running(Gamemap)
            agent_num = 2
        elif args.map == 'running-competition':
            map_id = random.randint(1, 10)
            Gamemap = create_scenario(args.map)
            game = Running_competition(meta_map=Gamemap, map_id=map_id)
            agent_num = 2
        elif args.map == 'seeks':
            game = Seeks(Gamemap)
            agent_num = 2
        elif args.map == 'table-hockey':
            game = table_hockey(Gamemap)
            agent_num = 2
        elif args.map == 'football':
            game = football(Gamemap)
            agent_num = 2
        elif args.map == 'wrestling':
            game = wrestling(Gamemap)
            agent_num = 2
        elif args.map == 'billiard':
            game = billiard(Gamemap)
            agent_num = 2
        elif args.map == 'billiard-competition':
            game = billiard_competition(Gamemap)
            agent_num = 2
        elif args.map == 'curling':
            game = curling(Gamemap)
            agent_num = 2
        elif args.map == 'curling-joint':
            game = curling_joint(Gamemap)
            agent_num = 2
        elif args.map == 'billiard-joint':
            game = billiard_joint(Gamemap)
            agent_num = 2
        elif args.map == 'curling-long':
            game = curling_long(Gamemap)
            agent_num = 2
        elif args.map == 'curling-competition':
            game = curling_competition(Gamemap)
            agent_num = 2
        elif args.map == 'all':
            game = AI_Olympics(random_selection=False, minimap=False)
            agent_num = 2
        elif args.map == 'all_v2':
            game = AI_Olympics(random_selection=False, minimap=False, vis=300, vis_clear=5)
            agent_num = 2

        agent = random_agent()
        rand_agent = random_agent()

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
                action1, action2 = agent.act(obs[0]), rand_agent.act(obs[1])
                action = [action1, action2]
            elif agent_num == 1:
                action1 = agent.act(obs)
                action = [action1]

            obs, reward, done, _ = game.step(action)
            print(f'reward = {reward}')

            if RENDER:
                game.render()  # normal render (no rgb_array)

                # Capture the screen manually
                screen = pygame.display.get_surface()
                if screen is not None:
                    img = pygame.surfarray.array3d(screen)
                    img = img.swapaxes(0, 1)  # Swap axes to match (height, width, channels)
                    frames.append(img)
                else:
                    print(f"Warning: Screen surface not available at step {step}")

        duration_t = time.time() - time_epi_s
        print("episode duration: ", duration_t,
              "step: ", step,
              "time-per-step:", (duration_t) / step)

        # Save gameplay if any frame was captured
        save_folder = '/Users/kuzeyarar/Desktop/Gameplay'
        os.makedirs(save_folder, exist_ok=True)
        gif_filename = f"{args.map}.gif"
        save_path = os.path.join(save_folder, gif_filename)

        if len(frames) > 0:
            imageio.mimsave(save_path, frames, fps=30)
            print("Saved successfully!")
        else:
            print("No frames captured. GIF not saved.")
