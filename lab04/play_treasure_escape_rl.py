from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from map_gen import generate_maps
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
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--generated-maps", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--difficulty", type=float, default=0.08)
    parser.add_argument("--path-width", type=int, default=1)
    parser.add_argument("--map-seed", type=int, default=11)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.05)
    return parser.parse_args()


def maps_from_args(args: argparse.Namespace) -> tuple[tuple[str, ...], ...] | None:
    if args.generated_maps <= 0:
        return None
    return generate_maps(
        count=args.generated_maps,
        width=args.width,
        height=args.height,
        difficulty=args.difficulty,
        path_width=args.path_width,
        seed=args.map_seed,
    )


def main() -> None:
    args = parse_args()
    maps = maps_from_args(args)
    agent = TrainedQAgent(args.q_table)
    env = TreasureEscapeEnv(
        render_mode="human" if args.render else None,
        map_id=args.map_id,
        max_steps=args.max_steps,
        maps=maps,
    )
    expected_shape = (env.map_count, env.height, env.width, 2, env.action_space.n)
    if agent.q_table.shape != expected_shape:
        raise ValueError(
            f"Q-table shape {agent.q_table.shape} does not match environment {expected_shape}. "
            "Use the same map source and generation parameters as during training."
        )
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
