from __future__ import annotations

from collections import deque
from enum import IntEnum
from collections.abc import Sequence
from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

try:
    import pygame
except ImportError:  # Rendering is optional until render_mode is used.
    pygame = None


class Actions(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


DEFAULT_MAPS = (
    (
        "##########",
        "#S.......#",
        "#.####X#.#",
        "#.#....#.#",
        "#.#.##.#T#",
        "#...#X.#.#",
        "###.#.##.#",
        "#.....#..#",
        "#X###...E#",
        "##########",
    ),
    (
        "##########",
        "#S....#..#",
        "#.###.#X.#",
        "#...#....#",
        "###.####.#",
        "#...#T...#",
        "#.#.###.##",
        "#.#...#..#",
        "#...#...E#",
        "##########",
    ),
    (
        "##########",
        "#S..#....#",
        "#.#.#.##.#",
        "#.#...#..#",
        "#.###.#T.#",
        "#...X.#..#",
        "#.###.##.#",
        "#...#....#",
        "#X#....E.#",
        "##########",
    ),
)


class TreasureEscapeEnv(gym.Env):
    """A Gymnasium grid game: collect the treasure and escape through the exit."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        render_mode: str | None = None,
        map_id: int | None = None,
        max_steps: int = 120,
        window_size: int = 600,
        maps: Sequence[Sequence[str]] | None = None,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.map_id = map_id
        self.max_steps = max_steps
        self.window_size = window_size
        self.maps = tuple(tuple(row for row in map_lines) for map_lines in (maps or DEFAULT_MAPS))
        if not self.maps:
            raise ValueError("At least one map is required.")

        self.height = len(self.maps[0])
        self.width = len(self.maps[0][0])
        self.size = self.height
        self.map_count = len(self.maps)
        self._validate_maps()
        self.observation_space = spaces.Dict(
            {
                "map_id": spaces.Discrete(self.map_count),
                "agent": spaces.Box(
                    low=np.array([0, 0], dtype=np.int64),
                    high=np.array([self.height - 1, self.width - 1], dtype=np.int64),
                    dtype=np.int64,
                ),
                "treasure": spaces.Box(
                    low=np.array([0, 0], dtype=np.int64),
                    high=np.array([self.height - 1, self.width - 1], dtype=np.int64),
                    dtype=np.int64,
                ),
                "exit": spaces.Box(
                    low=np.array([0, 0], dtype=np.int64),
                    high=np.array([self.height - 1, self.width - 1], dtype=np.int64),
                    dtype=np.int64,
                ),
                "has_treasure": spaces.Discrete(2),
                "walls": spaces.MultiBinary((self.height, self.width)),
                "traps": spaces.MultiBinary((self.height, self.width)),
            }
        )
        self.action_space = spaces.Discrete(len(Actions))

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([0, 1], dtype=np.int64),
            Actions.UP.value: np.array([-1, 0], dtype=np.int64),
            Actions.LEFT.value: np.array([0, -1], dtype=np.int64),
            Actions.DOWN.value: np.array([1, 0], dtype=np.int64),
        }

        self.window = None
        self.clock = None

        self._walls = np.zeros((self.height, self.width), dtype=np.int8)
        self._traps = np.zeros((self.height, self.width), dtype=np.int8)
        self._agent_location = np.zeros(2, dtype=np.int64)
        self._start_location = np.zeros(2, dtype=np.int64)
        self._treasure_location = np.zeros(2, dtype=np.int64)
        self._exit_location = np.zeros(2, dtype=np.int64)
        self._has_treasure = 0
        self._step_count = 0
        self._current_map_id = 0

    def _validate_maps(self) -> None:
        for map_lines in self.maps:
            if len(map_lines) != self.height:
                raise ValueError("All maps must have the same height.")
            for line in map_lines:
                if len(line) != self.width:
                    raise ValueError("All maps must have the same width.")

    def _parse_map(self, map_lines: tuple[str, ...]) -> None:
        self._walls.fill(0)
        self._traps.fill(0)
        start = treasure = exit_location = None

        for row, line in enumerate(map_lines):
            if len(line) != self.width:
                raise ValueError("All map rows must have the same length.")
            for col, tile in enumerate(line):
                position = np.array([row, col], dtype=np.int64)
                if tile == "#":
                    self._walls[row, col] = 1
                elif tile == "X":
                    self._traps[row, col] = 1
                elif tile == "S":
                    start = position
                elif tile == "T":
                    treasure = position
                elif tile == "E":
                    exit_location = position

        if start is None or treasure is None or exit_location is None:
            raise ValueError("Map must contain S, T and E tiles.")

        self._start_location = start
        self._agent_location = start.copy()
        self._treasure_location = treasure
        self._exit_location = exit_location

    def _get_obs(self) -> dict[str, np.ndarray | int]:
        return {
            "map_id": int(self._current_map_id),
            "agent": self._agent_location.copy(),
            "treasure": self._treasure_location.copy(),
            "exit": self._exit_location.copy(),
            "has_treasure": int(self._has_treasure),
            "walls": self._walls.copy(),
            "traps": self._traps.copy(),
        }

    def _shortest_path_length(self, start: np.ndarray, goal: np.ndarray) -> int | None:
        start_tuple = tuple(int(v) for v in start)
        goal_tuple = tuple(int(v) for v in goal)
        queue = deque([(start_tuple, 0)])
        visited = {start_tuple}

        while queue:
            current, distance = queue.popleft()
            if current == goal_tuple:
                return distance
            for direction in self._action_to_direction.values():
                nxt = (current[0] + int(direction[0]), current[1] + int(direction[1]))
                if (
                    nxt in visited
                    or not (0 <= nxt[0] < self.height and 0 <= nxt[1] < self.width)
                    or self._walls[nxt] == 1
                    or self._traps[nxt] == 1
                ):
                    continue
                visited.add(nxt)
                queue.append((nxt, distance + 1))
        return None

    def _get_info(self) -> dict[str, Any]:
        target = self._exit_location if self._has_treasure else self._treasure_location
        return {
            "map_id": int(self._current_map_id),
            "steps": self._step_count,
            "has_treasure": bool(self._has_treasure),
            "distance_to_current_target": self._shortest_path_length(
                self._agent_location, target
            ),
            "success": bool(
                self._has_treasure
                and np.array_equal(self._agent_location, self._exit_location)
            ),
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        requested_map_id = self.map_id
        if options and "map_id" in options:
            requested_map_id = int(options["map_id"])
        if requested_map_id is None:
            requested_map_id = int(self.np_random.integers(self.map_count))

        self._current_map_id = requested_map_id % self.map_count
        self._parse_map(self.maps[self._current_map_id])
        self._has_treasure = 0
        self._step_count = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: int):
        action = int(action)
        if action not in self._action_to_direction:
            raise ValueError(f"Unsupported action {action}.")

        self._step_count += 1
        reward = -0.01
        terminated = False
        truncated = False
        event = "move"

        direction = self._action_to_direction[action]
        new_location = self._agent_location + direction
        row, col = int(new_location[0]), int(new_location[1])

        blocked = (
            row < 0
            or row >= self.height
            or col < 0
            or col >= self.width
            or self._walls[row, col] == 1
        )

        if blocked:
            reward -= 0.2
            event = "blocked"
        else:
            self._agent_location = new_location

            if self._traps[row, col] == 1:
                reward -= 2.0
                terminated = True
                event = "trap"
            elif (
                not self._has_treasure
                and np.array_equal(self._agent_location, self._treasure_location)
            ):
                self._has_treasure = 1
                reward += 1.0
                event = "treasure"
            elif np.array_equal(self._agent_location, self._exit_location):
                if self._has_treasure:
                    reward += 5.0
                    terminated = True
                    event = "escaped"
                else:
                    reward -= 0.5
                    event = "locked_exit"

        if self._step_count >= self.max_steps and not terminated:
            truncated = True
            reward -= 1.0
            event = "timeout"

        observation = self._get_obs()
        info = self._get_info()
        info["event"] = event

        if self.render_mode == "human":
            self._render_frame()

        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "human":
            self._render_frame()
            return None
        return None

    def _render_frame(self):
        if pygame is None:
            raise DependencyNotInstalled(
                "pygame is required for rendering. Install it with: pip install pygame"
            )

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Treasure Escape")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((245, 246, 248))
        cell_width = self.window_size / self.width
        cell_height = self.window_size / self.height
        token_size = min(cell_width, cell_height)

        colors = {
            "floor": (245, 246, 248),
            "wall": (35, 40, 48),
            "trap": (190, 44, 44),
            "grid": (210, 214, 220),
            "treasure": (245, 184, 53),
            "exit_locked": (120, 130, 145),
            "exit_open": (41, 145, 88),
            "agent": (45, 102, 210),
        }

        for row in range(self.height):
            for col in range(self.width):
                rect = pygame.Rect(
                    round(col * cell_width),
                    round(row * cell_height),
                    round(cell_width),
                    round(cell_height),
                )
                if self._walls[row, col]:
                    pygame.draw.rect(canvas, colors["wall"], rect)
                elif self._traps[row, col]:
                    pygame.draw.rect(canvas, colors["trap"], rect)
                else:
                    pygame.draw.rect(canvas, colors["floor"], rect)
                pygame.draw.rect(canvas, colors["grid"], rect, 1)

        treasure_center = (
            round((self._treasure_location[1] + 0.5) * cell_width),
            round((self._treasure_location[0] + 0.5) * cell_height),
        )
        if not self._has_treasure:
            pygame.draw.circle(canvas, colors["treasure"], treasure_center, round(token_size * 0.28))

        exit_rect = pygame.Rect(
            round((self._exit_location[1] + 0.18) * cell_width),
            round((self._exit_location[0] + 0.18) * cell_height),
            round(cell_width * 0.64),
            round(cell_height * 0.64),
        )
        exit_color = colors["exit_open"] if self._has_treasure else colors["exit_locked"]
        pygame.draw.rect(canvas, exit_color, exit_rect, border_radius=6)

        agent_center = (
            round((self._agent_location[1] + 0.5) * cell_width),
            round((self._agent_location[0] + 0.5) * cell_height),
        )
        pygame.draw.circle(canvas, colors["agent"], agent_center, round(token_size * 0.3))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None and pygame is not None:
            pygame.display.quit()
            pygame.quit()
        self.window = None
        self.clock = None


def make_env(
    render_mode: str | None = None,
    map_id: int | None = None,
    maps: Sequence[Sequence[str]] | None = None,
) -> TreasureEscapeEnv:
    return TreasureEscapeEnv(render_mode=render_mode, map_id=map_id, maps=maps)
