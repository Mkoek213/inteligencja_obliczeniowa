# Projekt 4: Treasure Escape

Projekt zawiera własne środowisko Gymnasium, agenta RL uczonego metodą Q-learning, zapis krzywej uczenia oraz tryb graficzny.

## Pliki

- `treasure_escape_env.py` - środowisko `TreasureEscapeEnv`.
- `map_gen.py` - proceduralny generator rozwiązywalnych map.
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

Trening na konkretnej planszy:

```bash
.venv/bin/python lab04/train_treasure_escape_rl.py --episodes 1000 --eval-episodes 60 --map-id 2
```

Jeżeli `--map-id` nie jest podane, środowisko przy każdym `reset()` losuje jedną z wbudowanych plansz.

Trening na mapach generowanych proceduralnie:

```bash
.venv/bin/python lab04/train_treasure_escape_rl.py --episodes 3000 --eval-episodes 100 --generated-maps 5 --width 14 --height 14 --difficulty 0.12 --path-width 2 --map-seed 11
```

Opcje generatora:

- `--generated-maps` - liczba map generowanych na start programu; `0` zostawia mapy wbudowane,
- `--width`, `--height` - rozmiar planszy,
- `--difficulty` - gęstość pułapek dla wartości `0..1` albo dokładna liczba pułapek dla wartości większej niż `1`,
- `--path-width` - szerokość korytarzy labiryntu w kratkach,
- `--map-seed` - seed generatora map; przy odtwarzaniu wytrenowanego modelu trzeba użyć tego samego seeda i parametrów.

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

Na konkretnej planszy:

```bash
.venv/bin/python lab04/play_treasure_escape_rl.py --episodes 10 --map-id 1
```

Na mapach generowanych użyj tych samych parametrów generatora co przy treningu:

```bash
.venv/bin/python lab04/play_treasure_escape_rl.py --q-table lab04/outputs/q_learning/q_table.npy --episodes 30 --generated-maps 5 --width 14 --height 14 --difficulty 0.12 --path-width 2 --map-seed 11
```

## Agent strategiczny BFS jako baseline

```bash
.venv/bin/python lab04/solve_treasure_escape.py --episodes 30
```

Na konkretnej planszy:

```bash
.venv/bin/python lab04/solve_treasure_escape.py --episodes 30 --map-id 0
```

Na mapach generowanych:

```bash
.venv/bin/python lab04/solve_treasure_escape.py --episodes 30 --generated-maps 5 --width 14 --height 14 --difficulty 0.12 --path-width 2 --map-seed 11
```

Sam podgląd wygenerowanych map:

```bash
.venv/bin/python lab04/map_gen.py --count 3 --width 14 --height 14 --difficulty 0.12 --path-width 2 --seed 11
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
