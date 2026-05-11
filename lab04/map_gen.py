from __future__ import annotations

import argparse
from collections import deque
from typing import Iterable

import numpy as np


TileMap = tuple[str, ...]
Position = tuple[int, int]

DELTAS = ((0, 1), (-1, 0), (0, -1), (1, 0))


def generate_map(
    width: int,
    height: int,
    difficulty: float = 0.08,
    path_width: int = 1,
    seed: int | None = None,
    max_attempts: int = 100,
) -> TileMap:
    """Generate one solvable Treasure Escape map.

    ``difficulty`` is interpreted as trap density for values from 0 to 1 and as
    an exact trap count for values greater than 1.
    """
    if width < 5 or height < 5:
        raise ValueError("Map width and height must be at least 5.")
    if path_width < 1:
        raise ValueError("Path width must be at least 1.")
    if path_width > min(width, height) - 2:
        raise ValueError("Path width must fit inside the outer wall border.")
    if difficulty < 0:
        raise ValueError("Difficulty cannot be negative.")

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        grid = _carve_maze(width, height, path_width, rng)
        start, treasure, exit_location = _choose_key_positions(grid, rng)
        _write_specials(grid, start, treasure, exit_location)

        protected = set(_shortest_path(grid, start, treasure) or [])
        protected.update(_shortest_path(grid, treasure, exit_location) or [])
        _place_traps(grid, difficulty, protected, rng)

        map_lines = _grid_to_lines(grid)
        if has_solution(map_lines):
            return map_lines

    raise RuntimeError("Failed to generate a solvable map. Try lower difficulty.")


def generate_maps(
    count: int,
    width: int,
    height: int,
    difficulty: float = 0.08,
    path_width: int = 1,
    seed: int | None = None,
) -> tuple[TileMap, ...]:
    if count < 1:
        raise ValueError("Map count must be at least 1.")
    rng = np.random.default_rng(seed)
    return tuple(
        generate_map(
            width=width,
            height=height,
            difficulty=difficulty,
            path_width=path_width,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        for _ in range(count)
    )


def has_solution(map_lines: Iterable[str]) -> bool:
    grid = np.array([list(row) for row in map_lines])
    start = _find_tile(grid, "S")
    treasure = _find_tile(grid, "T")
    exit_location = _find_tile(grid, "E")
    return (
        start is not None
        and treasure is not None
        and exit_location is not None
        and _shortest_path(grid, start, treasure) is not None
        and _shortest_path(grid, treasure, exit_location) is not None
    )


def _carve_maze(
    width: int, height: int, path_width: int, rng: np.random.Generator
) -> np.ndarray:
    grid = np.full((height, width), "#", dtype="<U1")
    row_starts = _corridor_starts(height, path_width)
    col_starts = _corridor_starts(width, path_width)
    if len(row_starts) < 2 or len(col_starts) < 2:
        raise ValueError("Map is too small for the selected path width.")

    start_cell = (
        int(rng.integers(len(row_starts))),
        int(rng.integers(len(col_starts))),
    )
    visited = {start_cell}
    stack = [start_cell]
    _carve_block(grid, row_starts[start_cell[0]], col_starts[start_cell[1]], path_width)

    while stack:
        row_idx, col_idx = stack[-1]
        candidates = [
            (row_idx + dr, col_idx + dc)
            for dr, dc in DELTAS
            if 0 <= row_idx + dr < len(row_starts)
            and 0 <= col_idx + dc < len(col_starts)
            and (row_idx + dr, col_idx + dc) not in visited
        ]
        if not candidates:
            stack.pop()
            continue

        next_cell = candidates[int(rng.integers(len(candidates)))]
        visited.add(next_cell)
        stack.append(next_cell)
        _carve_between(
            grid,
            (row_starts[row_idx], col_starts[col_idx]),
            (row_starts[next_cell[0]], col_starts[next_cell[1]]),
            path_width,
        )

    return grid


def _corridor_starts(length: int, path_width: int) -> list[int]:
    spacing = path_width + 1
    return list(range(1, length - path_width, spacing))


def _carve_block(grid: np.ndarray, row: int, col: int, path_width: int) -> None:
    grid[row : row + path_width, col : col + path_width] = "."


def _carve_between(
    grid: np.ndarray, first: Position, second: Position, path_width: int
) -> None:
    row_a, col_a = first
    row_b, col_b = second
    row_min, row_max = sorted((row_a, row_b))
    col_min, col_max = sorted((col_a, col_b))
    grid[row_min : row_max + path_width, col_min : col_max + path_width] = "."


def _choose_key_positions(
    grid: np.ndarray, rng: np.random.Generator
) -> tuple[Position, Position, Position]:
    floors = _floor_positions(grid)
    probe = floors[int(rng.integers(len(floors)))]
    start = _farthest_position(grid, probe)
    exit_location = _farthest_position(grid, start)
    distances_from_start = _distances(grid, start)
    distances_from_exit = _distances(grid, exit_location)
    treasure = max(
        floors,
        key=lambda pos: min(
            distances_from_start.get(pos, -1),
            distances_from_exit.get(pos, -1),
        ),
    )
    return start, treasure, exit_location


def _write_specials(
    grid: np.ndarray, start: Position, treasure: Position, exit_location: Position
) -> None:
    grid[start] = "S"
    grid[treasure] = "T"
    grid[exit_location] = "E"


def _place_traps(
    grid: np.ndarray,
    difficulty: float,
    protected: set[Position],
    rng: np.random.Generator,
) -> None:
    candidates = [
        pos
        for pos in _floor_positions(grid)
        if pos not in protected and grid[pos] == "."
    ]
    if difficulty <= 1:
        trap_count = int(round(len(candidates) * difficulty))
    else:
        trap_count = int(round(difficulty))
    trap_count = max(0, min(trap_count, len(candidates)))

    if trap_count == 0:
        return

    order = rng.permutation(len(candidates))
    for idx in order[:trap_count]:
        grid[candidates[int(idx)]] = "X"


def _shortest_path(
    grid: np.ndarray, start: Position, goal: Position
) -> list[Position] | None:
    queue = deque([start])
    previous: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in _neighbors(grid, current):
            if neighbor in previous:
                continue
            previous[neighbor] = current
            queue.append(neighbor)

    if goal not in previous:
        return None

    path = []
    current: Position | None = goal
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


def _distances(grid: np.ndarray, start: Position) -> dict[Position, int]:
    queue = deque([(start, 0)])
    distances = {start: 0}
    while queue:
        current, distance = queue.popleft()
        for neighbor in _neighbors(grid, current):
            if neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append((neighbor, distance + 1))
    return distances


def _neighbors(grid: np.ndarray, position: Position) -> Iterable[Position]:
    rows, cols = grid.shape
    for dr, dc in DELTAS:
        nxt = (position[0] + dr, position[1] + dc)
        if (
            0 <= nxt[0] < rows
            and 0 <= nxt[1] < cols
            and grid[nxt] != "#"
            and grid[nxt] != "X"
        ):
            yield nxt


def _farthest_position(grid: np.ndarray, start: Position) -> Position:
    distances = _distances(grid, start)
    return max(distances, key=distances.__getitem__)


def _floor_positions(grid: np.ndarray) -> list[Position]:
    rows, cols = grid.shape
    return [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if grid[row, col] != "#"
    ]


def _find_tile(grid: np.ndarray, tile: str) -> Position | None:
    matches = np.argwhere(grid == tile)
    if len(matches) == 0:
        return None
    row, col = matches[0]
    return int(row), int(col)


def _grid_to_lines(grid: np.ndarray) -> TileMap:
    return tuple("".join(row.tolist()) for row in grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Treasure Escape maps.")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--difficulty", type=float, default=0.08)
    parser.add_argument("--path-width", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maps = generate_maps(
        count=args.count,
        width=args.width,
        height=args.height,
        difficulty=args.difficulty,
        path_width=args.path_width,
        seed=args.seed,
    )
    for map_id, map_lines in enumerate(maps):
        if len(maps) > 1:
            print(f"map {map_id}:")
        print("\n".join(map_lines))
        if map_id != len(maps) - 1:
            print()


if __name__ == "__main__":
    main()
