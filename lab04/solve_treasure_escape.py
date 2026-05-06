from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from treasure_escape_env import Actions, TreasureEscapeEnv


ACTION_TO_DELTA = {
    Actions.RIGHT.value: (0, 1),
    Actions.UP.value: (-1, 0),
    Actions.LEFT.value: (0, -1),
    Actions.DOWN.value: (1, 0),
}


@dataclass
class EpisodeResult:
    success: bool
    total_reward: float
    steps: int
    events: list[str]


class StrategicBFSAgent:
    """Agent that plans a shortest safe route to treasure, then to the exit."""

    def choose_action(self, observation: dict) -> int:
        agent = tuple(int(v) for v in observation["agent"])
        treasure = tuple(int(v) for v in observation["treasure"])
        exit_location = tuple(int(v) for v in observation["exit"])
        has_treasure = bool(observation["has_treasure"])
        target = exit_location if has_treasure else treasure

        path = self._shortest_path(
            start=agent,
            goal=target,
            walls=observation["walls"],
            traps=observation["traps"],
        )
        if len(path) >= 2:
            return self._action_between(path[0], path[1])

        return self._first_valid_action(agent, observation["walls"], observation["traps"])

    def _shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        walls: np.ndarray,
        traps: np.ndarray,
    ) -> list[tuple[int, int]]:
        queue = deque([start])
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for neighbor in self._neighbors(current, walls, traps):
                if neighbor in previous:
                    continue
                previous[neighbor] = current
                queue.append(neighbor)

        if goal not in previous:
            return []

        path = []
        current: tuple[int, int] | None = goal
        while current is not None:
            path.append(current)
            current = previous[current]
        return list(reversed(path))

    def _neighbors(
        self, position: tuple[int, int], walls: np.ndarray, traps: np.ndarray
    ) -> Iterable[tuple[int, int]]:
        rows, cols = walls.shape
        for delta in ACTION_TO_DELTA.values():
            nxt = (position[0] + delta[0], position[1] + delta[1])
            if (
                0 <= nxt[0] < rows
                and 0 <= nxt[1] < cols
                and walls[nxt] == 0
                and traps[nxt] == 0
            ):
                yield nxt

    def _action_between(self, current: tuple[int, int], nxt: tuple[int, int]) -> int:
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        for action, action_delta in ACTION_TO_DELTA.items():
            if delta == action_delta:
                return action
        raise ValueError(f"Cells are not adjacent: {current} -> {nxt}")

    def _first_valid_action(
        self, agent: tuple[int, int], walls: np.ndarray, traps: np.ndarray
    ) -> int:
        for action, delta in ACTION_TO_DELTA.items():
            nxt = (agent[0] + delta[0], agent[1] + delta[1])
            if (
                0 <= nxt[0] < walls.shape[0]
                and 0 <= nxt[1] < walls.shape[1]
                and walls[nxt] == 0
                and traps[nxt] == 0
            ):
                return action
        return Actions.RIGHT.value


class RandomAgent:
    def __init__(self, action_space, seed: int | None = None):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def choose_action(self, observation: dict) -> int:
        return int(self.rng.integers(self.action_space.n))


def run_episode(
    env: TreasureEscapeEnv,
    agent,
    seed: int | None = None,
    render: bool = False,
    sleep: float = 0.0,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    events: list[str] = []

    while True:
        action = agent.choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        events.append(info["event"])

        if render:
            env.render()
            if sleep > 0:
                time.sleep(sleep)

        if terminated or truncated:
            return EpisodeResult(
                success=bool(info["success"]),
                total_reward=total_reward,
                steps=int(info["steps"]),
                events=events,
            )


def summarize_results(name: str, results: list[EpisodeResult]) -> None:
    successes = sum(result.success for result in results)
    avg_reward = sum(result.total_reward for result in results) / len(results)
    avg_steps = sum(result.steps for result in results) / len(results)
    print(f"{name}:")
    print(f"  episodes:      {len(results)}")
    print(f"  success rate:  {successes / len(results):.1%}")
    print(f"  avg reward:    {avg_reward:.3f}")
    print(f"  avg steps:     {avg_steps:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agents in TreasureEscapeEnv.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--map-id", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--random-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "human" if args.render else None

    strategic_env = TreasureEscapeEnv(render_mode=render_mode, map_id=args.map_id)
    strategic_agent = StrategicBFSAgent()
    strategic_results = [
        run_episode(
            strategic_env,
            strategic_agent,
            seed=args.seed + episode,
            render=args.render,
            sleep=args.sleep,
        )
        for episode in range(args.episodes)
    ]
    strategic_env.close()
    summarize_results("Strategic BFS agent", strategic_results)

    if args.random_baseline:
        random_env = TreasureEscapeEnv(map_id=args.map_id)
        random_agent = RandomAgent(random_env.action_space, seed=args.seed)
        random_results = [
            run_episode(random_env, random_agent, seed=args.seed + episode)
            for episode in range(args.episodes)
        ]
        random_env.close()
        summarize_results("Random baseline", random_results)


if __name__ == "__main__":
    main()
