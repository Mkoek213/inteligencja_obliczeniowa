from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from treasure_escape_env import TreasureEscapeEnv


class TrainedQAgent:
    def __init__(self, q_table_path: Path):
        self.q_table = np.load(q_table_path)

    def choose_action(self, observation: dict) -> int:
        row, col = (int(v) for v in observation["agent"])
        state = (
            int(observation["map_id"]),
            row,
            col,
            int(observation["has_treasure"]),
        )
        return int(np.argmax(self.q_table[state]))


def run_episode(
    env: TreasureEscapeEnv,
    agent: TrainedQAgent,
    seed: int,
    render: bool,
    sleep: float,
) -> tuple[bool, float, int]:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            env.render()
            time.sleep(sleep)

    return bool(info["success"]), total_reward, int(info["steps"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Treasure Escape with a trained Q-table.")
    parser.add_argument(
        "--q-table",
        type=Path,
        default=Path("lab04/outputs/q_learning/q_table.npy"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--map-id", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = TrainedQAgent(args.q_table)
    env = TreasureEscapeEnv(render_mode="human" if args.render else None, map_id=args.map_id)
    results = [
        run_episode(
            env,
            agent,
            seed=args.seed + episode,
            render=args.render,
            sleep=args.sleep,
        )
        for episode in range(args.episodes)
    ]
    env.close()

    successes = [int(result[0]) for result in results]
    rewards = [result[1] for result in results]
    steps = [result[2] for result in results]
    print("Trained Q-learning agent:")
    print(f"  episodes:      {args.episodes}")
    print(f"  success rate:  {np.mean(successes):.1%}")
    print(f"  avg reward:    {np.mean(rewards):.3f}")
    print(f"  avg steps:     {np.mean(steps):.1f}")


if __name__ == "__main__":
    main()
