# Projekt 4: Treasure Escape

Projekt zawiera własne środowisko Gymnasium, agenta RL uczonego metodą Q-learning, zapis krzywej uczenia oraz tryb graficzny.

## Pliki

- `treasure_escape_env.py` - środowisko `TreasureEscapeEnv`.
- `train_treasure_escape_rl.py` - trening Q-learning, ewaluacja i zapis krzywej uczenia.
- `solve_treasure_escape.py` - agent strategiczny BFS i eksperymenty.
- `sprawozdanie.md` - opis środowiska, algorytmu i wyników.
- `environment_creation.py` - pomocniczy tutorial Gymnasium z treści zadania.

## Instalacja zależności

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Jeżeli zależności są już zainstalowane w aktywnym środowisku Pythona, wystarczy:

```bash
python3 -m pip install -r requirements.txt
```

## Trening agenta RL

```bash
.venv/bin/python lab04/train_treasure_escape_rl.py --episodes 1000 --eval-episodes 60
```

Skrypt zapisuje:

- `lab04/outputs/q_learning/q_table.npy` - wyuczona tablica Q,
- `lab04/outputs/q_learning/learning_curve.csv` - dane krzywej uczenia,
- `lab04/outputs/q_learning/learning_curve.png` - wykres krzywej uczenia.

## Uruchomienie wytrenowanego agenta RL

```bash
.venv/bin/python lab04/play_treasure_escape_rl.py --episodes 30
```

Z renderingiem:

```bash
.venv/bin/python lab04/play_treasure_escape_rl.py --episodes 1 --render
```

## Agent strategiczny BFS jako baseline

```bash
.venv/bin/python lab04/solve_treasure_escape.py --episodes 30
```

## Porównanie z agentem losowym

```bash
.venv/bin/python lab04/solve_treasure_escape.py --episodes 100 --random-baseline
```

## Tryb graficzny

```bash
.venv/bin/python lab04/solve_treasure_escape.py --episodes 1 --render
```

## Sprawdzenie zgodności z Gymnasium

```bash
.venv/bin/python -c "from gymnasium.utils.env_checker import check_env; from lab04.treasure_escape_env import TreasureEscapeEnv; check_env(TreasureEscapeEnv())"
```
