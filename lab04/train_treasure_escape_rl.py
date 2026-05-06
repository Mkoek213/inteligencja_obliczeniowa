from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from treasure_escape_env import DEFAULT_MAPS, TreasureEscapeEnv


@dataclass
class TrainingConfig:
    episodes: int = 5000
    alpha: float = 0.18
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay: float = 0.996
    shaping_scale: float = 0.03
    seed: int = 11


class QLearningAgent:
    def __init__(
        self,
        q_table: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
    ):
        self.q_table = q_table
        self.epsilon = epsilon
        self.rng = rng

    @classmethod
    def fresh(cls, env: TreasureEscapeEnv, epsilon: float, rng: np.random.Generator):
        q_shape = (
            len(DEFAULT_MAPS),
            env.size,
            env.size,
            2,
            env.action_space.n,
        )
        return cls(np.zeros(q_shape, dtype=np.float64), epsilon, rng)

    @classmethod
    def load(cls, path: Path, epsilon: float, rng: np.random.Generator):
        return cls(np.load(path), epsilon, rng)

    def state_index(self, observation: dict) -> tuple[int, int, int, int]:
        row, col = (int(v) for v in observation["agent"])
        return (
            int(observation["map_id"]),
            row,
            col,
            int(observation["has_treasure"]),
        )

    def choose_action(self, observation: dict, explore: bool = True) -> int:
        state = self.state_index(observation)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q_table.shape[-1]))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        done: bool,
        alpha: float,
        gamma: float,
    ) -> None:
        state = self.state_index(observation)
        next_state = self.state_index(next_observation)
        current_q = self.q_table[state + (action,)]
        next_best_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        target_q = reward + gamma * next_best_q
        self.q_table[state + (action,)] = current_q + alpha * (target_q - current_q)


def shaped_reward(
    reward: float,
    previous_info: dict,
    next_info: dict,
    shaping_scale: float,
) -> float:
    previous_distance = previous_info["distance_to_current_target"]
    next_distance = next_info["distance_to_current_target"]
    if previous_distance is None or next_distance is None:
        return reward
    return reward + shaping_scale * (previous_distance - next_distance)


def train(config: TrainingConfig, output_dir: Path) -> tuple[QLearningAgent, list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    env = TreasureEscapeEnv()
    agent = QLearningAgent.fresh(env, epsilon=config.epsilon_start, rng=rng)
    history: list[dict] = []

    for episode in range(1, config.episodes + 1):
        observation, info = env.reset(seed=config.seed + episode)
        total_env_reward = 0.0
        total_training_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.choose_action(observation, explore=True)
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            train_reward = shaped_reward(
                reward,
                info,
                next_info,
                shaping_scale=config.shaping_scale,
            )
            agent.update(
                observation,
                action,
                train_reward,
                next_observation,
                done,
                alpha=config.alpha,
                gamma=config.gamma,
            )

            observation = next_observation
            info = next_info
            total_env_reward += reward
            total_training_reward += train_reward

        history.append(
            {
                "episode": episode,
                "reward": total_env_reward,
                "training_reward": total_training_reward,
                "steps": int(info["steps"]),
                "success": int(info["success"]),
                "epsilon": agent.epsilon,
            }
        )
        agent.epsilon = max(config.epsilon_end, agent.epsilon * config.epsilon_decay)

    env.close()
    save_outputs(agent, history, output_dir)
    return agent, history


def evaluate(
    agent: QLearningAgent,
    episodes: int,
    seed: int,
    map_id: int | None = None,
    render: bool = False,
    sleep: float = 0.04,
) -> dict[str, float]:
    env = TreasureEscapeEnv(render_mode="human" if render else None, map_id=map_id)
    rewards = []
    steps = []
    successes = []

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.choose_action(observation, explore=False)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if render:
                env.render()
                time.sleep(sleep)

        rewards.append(total_reward)
        steps.append(int(info["steps"]))
        successes.append(int(info["success"]))

    env.close()
    return {
        "episodes": float(episodes),
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps)),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_outputs(agent: QLearningAgent, history: list[dict], output_dir: Path) -> None:
    np.save(output_dir / "q_table.npy", agent.q_table)

    csv_path = output_dir / "learning_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    episodes = np.array([row["episode"] for row in history], dtype=np.int64)
    rewards = np.array([row["reward"] for row in history], dtype=np.float64)
    success = np.array([row["success"] for row in history], dtype=np.float64)
    window = min(100, max(1, len(history) // 10))
    reward_ma = moving_average(rewards, window)
    success_ma = moving_average(success, window)
    ma_episodes = episodes[window - 1 :]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(episodes, rewards, color="0.78", linewidth=0.7, label="reward")
    axes[0].plot(ma_episodes, reward_ma, color="#1f77b4", linewidth=2, label=f"{window}-episode average")
    axes[0].set_ylabel("Episode reward")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(ma_episodes, success_ma, color="#2ca02c", linewidth=2)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Treasure Escape Q-learning curve")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, default=Path("lab04/outputs/q_learning"))
    parser.add_argument("--render-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(episodes=args.episodes, seed=args.seed)
    agent, history = train(config, args.output_dir)
    metrics = evaluate(agent, episodes=args.eval_episodes, seed=args.seed + 100_000)

    print("Q-learning training finished")
    print(f"  episodes:       {args.episodes}")
    print(f"  final epsilon:  {history[-1]['epsilon']:.3f}")
    print(f"  q-table:        {args.output_dir / 'q_table.npy'}")
    print(f"  curve csv:      {args.output_dir / 'learning_curve.csv'}")
    print(f"  curve png:      {args.output_dir / 'learning_curve.png'}")
    print("Evaluation without exploration:")
    print(f"  episodes:       {int(metrics['episodes'])}")
    print(f"  success rate:   {metrics['success_rate']:.1%}")
    print(f"  avg reward:     {metrics['avg_reward']:.3f}")
    print(f"  avg steps:      {metrics['avg_steps']:.1f}")

    if args.render_eval:
        eval_agent = QLearningAgent(agent.q_table, epsilon=0.0, rng=np.random.default_rng(args.seed))
        evaluate(eval_agent, episodes=1, seed=args.seed + 200_000, render=True)


if __name__ == "__main__":
    main()
