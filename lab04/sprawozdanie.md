# Projekt 4: Własne środowisko Gymnasium - Treasure Escape

## 1. Cel projektu

Celem projektu było przygotowanie własnego środowiska zgodnego z biblioteką Gymnasium, napisanie agenta RL rozwiązującego problem, zapisanie krzywej uczenia oraz dodanie trybu graficznego. Zaimplementowana gra nosi nazwę `Treasure Escape`. Agent znajduje się w labiryncie, musi zebrać skarb, a następnie dotrzeć do wyjścia. Po drodze powinien omijać ściany oraz pola-pułapki.

Projekt spełnia wymagania zadania na najwyższy próg, ponieważ zawiera własne środowisko Gymnasium, program uczący agenta metodą reinforcement learning, dużą dyskretną przestrzeń obserwacji, zapis krzywej uczenia oraz rendering graficzny w `pygame`.

## 2. Opis środowiska

Środowisko zostało zaimplementowane w pliku `treasure_escape_env.py` jako klasa `TreasureEscapeEnv`, która dziedziczy po `gymnasium.Env`. Plansza ma rozmiar `10x10` i jest reprezentowana przez kafelki:

- `S` - start agenta,
- `T` - skarb,
- `E` - wyjście,
- `#` - ściana,
- `X` - pułapka,
- `.` - wolne pole.

W środowisku dostępne są trzy warianty map wbudowanych. Przy resecie wybierana jest jedna z nich, chyba że użytkownik poda konkretny `map_id`. Dzięki temu agent jest sprawdzany na kilku układach labiryntu, ale mapy pozostają deterministyczne i możliwe do rozwiązania. Środowisko obsługuje też listę map przekazaną z zewnątrz, co pozwala trenować i uruchamiać agentów na mapach generowanych proceduralnie.

Przestrzeń akcji jest dyskretna i zawiera cztery akcje:

- `RIGHT = 0`,
- `UP = 1`,
- `LEFT = 2`,
- `DOWN = 3`.

Przestrzeń obserwacji jest słownikiem `spaces.Dict`. Zawiera identyfikator mapy, pozycję agenta, pozycję skarbu, pozycję wyjścia, informację o posiadaniu skarbu, macierz ścian oraz macierz pułapek. Jest to duża przestrzeń dyskretna, ponieważ stan środowiska obejmuje nie tylko pozycję agenta, lecz także strukturę całej planszy.

## 2.1. Generator map

Generator został dodany w pliku `map_gen.py`. Główna funkcja `generate_map(width, height, difficulty, path_width, seed)` tworzy najpierw labirynt korytarzy, a dopiero potem rozmieszcza elementy gry. Labirynt powstaje algorytmem randomized DFS/backtracking na siatce komórek pomocniczych. Każda odwiedzona komórka oraz przejście między komórkami są wycinane w macierzy ścian jako korytarz o szerokości `path_width` kratek.

Po utworzeniu labiryntu generator wybiera pozycje `S`, `T` i `E` na podstawie odległości BFS. Start i wyjście są umieszczane daleko od siebie, a skarb trafia w punkt oddalony zarówno od startu, jak i wyjścia. Dzięki temu trasa ma sensowną długość i wymusza cel etapowy.

Parametr `difficulty` steruje liczbą pułapek `X`. Dla wartości od `0` do `1` jest traktowany jako gęstość pułapek na dostępnych polach, a dla wartości większych niż `1` jako dokładna liczba pułapek. Generator najpierw wyznacza chronioną ścieżkę `S -> T -> E`, nie umieszcza na niej pułapek, a następnie dodatkowo sprawdza całą mapę algorytmem BFS. Jeżeli mapa nie ma rozwiązania, generowanie jest ponawiane.

Przykładowa komenda podglądu map:

```bash
.venv/bin/python lab04/map_gen.py --count 3 --width 14 --height 14 --difficulty 0.12 --path-width 2 --seed 11
```

## 3. System nagród

Agent otrzymuje nagrody zaprojektowane tak, aby promować szybkie i poprawne ukończenie gry:

- `-0.01` za każdy krok,
- `-0.2` za próbę wejścia w ścianę,
- `+1.0` za zebranie skarbu,
- `+5.0` za dotarcie do wyjścia ze skarbem,
- `-0.5` za próbę użycia wyjścia bez skarbu,
- `-2.0` za wejście na pułapkę,
- `-1.0` za przekroczenie limitu kroków.

Epizod kończy się sukcesem, gdy agent posiada skarb i dotrze do wyjścia. Epizod kończy się porażką, gdy agent wejdzie na pułapkę albo przekroczy limit kroków.

## 4. Agent RL rozwiązujący problem

Agent RL został zaimplementowany w pliku `train_treasure_escape_rl.py`. Wykorzystano tablicowy algorytm Q-learning. Stan używany przez agenta ma postać:

```text
(map_id, agent_row, agent_col, has_treasure)
```

Tablica Q ma więc wymiary:

```text
liczba_map x liczba_wierszy x liczba_kolumn x 2 x liczba_akcji
```

W każdej iteracji agent wybiera akcję strategią epsilon-greedy. Na początku treningu `epsilon` jest wysokie, więc agent często eksploruje. W kolejnych epizodach `epsilon` maleje, dzięki czemu agent coraz częściej wykorzystuje najlepszą znaną akcję. Aktualizacja Q ma standardową postać:

```text
Q(s, a) = Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

W treningu zastosowano niewielkie kształtowanie nagrody na podstawie zmiany odległości do aktualnego celu. Jeżeli agent zbliża się do skarbu albo po zebraniu skarbu do wyjścia, otrzymuje lekko większy sygnał treningowy. Nagroda środowiska pozostaje raportowana osobno jako wynik epizodu.

## 5. Baseline strategiczny

Agent został zaimplementowany w pliku `solve_treasure_escape.py` jako `StrategicBFSAgent`. Nie wykonuje akcji losowo. W każdej turze wyznacza najkrótszą bezpieczną ścieżkę za pomocą przeszukiwania wszerz BFS:

1. Jeżeli agent nie ma skarbu, celem jest pole `T`.
2. Jeżeli agent zebrał skarb, celem jest pole `E`.
3. BFS ignoruje pola będące ścianami oraz pułapkami.
4. Agent wykonuje pierwszą akcję z wyznaczonej najkrótszej ścieżki.

Taka strategia jest wystarczająca dla planszy kratowej z równymi kosztami przejścia. BFS gwarantuje znalezienie najkrótszej ścieżki, jeśli taka ścieżka istnieje.

## 6. Eksperymenty

Trening agenta RL na mapach wbudowanych uruchomiono komendą:

```bash
.venv/bin/python lab04/train_treasure_escape_rl.py --episodes 1000 --eval-episodes 60
```

Trening na mapach generowanych uruchamia się przez podanie liczby map i parametrów generatora:

```bash
.venv/bin/python lab04/train_treasure_escape_rl.py --episodes 3000 --eval-episodes 100 --generated-maps 5 --width 14 --height 14 --difficulty 0.12 --path-width 2 --map-seed 11
```

W przypadku map generowanych tablica Q pozostaje tablicowa i ma wymiary zależne od aktywnego środowiska:

```text
liczba_wygenerowanych_map x wysokość x szerokość x 2 x liczba_akcji
```

Do odtworzenia wytrenowanego modelu na mapach generowanych trzeba podać te same parametry generatora i ten sam `--map-seed`, ponieważ `map_id` odnosi się do kolejności map w wygenerowanym zestawie.

Skrypt zapisuje wyniki do katalogu `lab04/outputs/q_learning`:

- `q_table.npy` - wyuczona tablica Q,
- `learning_curve.csv` - nagroda, liczba kroków, sukces i epsilon dla każdego epizodu,
- `learning_curve.png` - wykres krzywej uczenia.

Po treningu wyuczonego agenta można uruchomić bez ponownego uczenia:

```bash
.venv/bin/python lab04/play_treasure_escape_rl.py --episodes 30
```

Przykładowy wynik ewaluacji po treningu:

| Agent | Epizody ewaluacji | Skuteczność | Średnia nagroda | Średnia liczba kroków |
| --- | ---: | ---: | ---: | ---: |
| Q-learning | 60 | 100.0% | 5.815 | 18.5 |

Krzywa uczenia pokazuje wzrost skuteczności wraz ze spadkiem eksploracji. Początkowo agent często trafia na ściany, pułapki lub przekracza limit kroków. Po treningu wybiera akcje prowadzące najpierw do skarbu, a następnie do wyjścia.

Do eksperymentów przygotowano uruchomienie z poziomu terminala. Przykładowa komenda:

```bash
.venv/bin/python lab04/solve_treasure_escape.py --episodes 100 --random-baseline
```

Program uruchamia agenta strategicznego oraz opcjonalnie agenta losowego. Dla każdego agenta raportowane są:

- liczba epizodów,
- skuteczność ukończenia gry,
- średnia suma nagród,
- średnia liczba kroków.

Oczekiwany wynik jest taki, że agent BFS osiąga skuteczność bliską `100%`, ponieważ zna obserwację planszy i planuje trasę. Agent losowy powinien mieć znacznie gorszą skuteczność, ponieważ często uderza w ściany, wchodzi na pułapki albo nie kończy gry przed limitem kroków.

Przykładowe wyniki uzyskane komendą:

```bash
.venv/bin/python lab04/solve_treasure_escape.py --episodes 10 --random-baseline
```

| Agent | Epizody | Skuteczność | Średnia nagroda | Średnia liczba kroków |
| --- | ---: | ---: | ---: | ---: |
| Strategic BFS agent | 10 | 100.0% | 5.807 | 19.3 |
| Random baseline | 10 | 0.0% | -8.782 | 72.2 |

Sprawdzono również każdą z trzech map oddzielnie. Agent strategiczny uzyskał skuteczność `100%` dla `map-id` równego `0`, `1` oraz `2`.

## 7. Tryb graficzny

Rendering został przygotowany w `pygame`. Środowisko obsługuje tryby `human` oraz `rgb_array`, zgodnie z konwencją Gymnasium. W trybie graficznym widoczne są ściany, pułapki, skarb, wyjście oraz agent. Po zebraniu skarbu kolor wyjścia zmienia się z zablokowanego na otwarte.

Uruchomienie trybu graficznego:

```bash
.venv/bin/python lab04/solve_treasure_escape.py --episodes 1 --render
```

## 8. Wnioski

Zaimplementowane środowisko pokazuje kompletny problem decyzyjny w stylu Gymnasium. Agent musi realizować cel etapowy: najpierw zdobyć skarb, a dopiero potem dotrzeć do wyjścia. Trening Q-learning pozwala pokazać krzywą uczenia i poprawę jakości decyzji w kolejnych epizodach. Agent BFS został zostawiony jako dodatkowy baseline pokazujący optymalną strategię planistyczną dla tej planszy.
