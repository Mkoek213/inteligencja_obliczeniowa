# Opis gry Treasure Escape

## Czym jest gra

`Treasure Escape` to własne środowisko zgodne z biblioteką Gymnasium. Gra odbywa się na dwuwymiarowej planszy kratowej. Agent porusza się po labiryncie, musi zebrać skarb, a następnie dotrzeć do wyjścia. Plansze mogą pochodzić z zestawu wbudowanego albo być generowane proceduralnie.

Plansza składa się z kafelków:

- `S` - pozycja startowa agenta,
- `T` - skarb,
- `E` - wyjście,
- `#` - ściana,
- `X` - pułapka,
- `.` - puste pole, po którym można przejść.

Wbudowane mapy mają rozmiar `10x10`. Generator map pozwala tworzyć większe mapy prostokątne, np. `100x100`, z regulowaną szerokością korytarzy i poziomem trudności.

## Zasady gry

Agent wykonuje w każdej turze jedną z czterech akcji:

- `RIGHT = 0` - ruch w prawo,
- `UP = 1` - ruch w górę,
- `LEFT = 2` - ruch w lewo,
- `DOWN = 3` - ruch w dół.

Ruch poza planszę albo w ścianę jest blokowany. Agent pozostaje wtedy na obecnym polu i dostaje karę. Wejście na pułapkę kończy epizod porażką. Samo dojście do wyjścia nie wystarcza, jeżeli agent nie zebrał jeszcze skarbu. Sukces następuje dopiero wtedy, gdy agent najpierw wejdzie na pole `T`, a potem dotrze do `E`.

Epizod kończy się w trzech sytuacjach:

- agent zebrał skarb i dotarł do wyjścia,
- agent wszedł na pułapkę,
- agent przekroczył limit kroków `max_steps`.

## Cel gry

Celem agenta jest znalezienie bezpiecznej i możliwie krótkiej trasy:

```text
start -> skarb -> wyjście
```

Agent powinien unikać ścian, pułapek i zbędnych ruchów. Problem jest etapowy, ponieważ cel zmienia się w trakcie epizodu: przed zebraniem skarbu celem jest `T`, a po zebraniu skarbu celem jest `E`.

## System nagród

Środowisko zwraca nagrody, które premiują ukończenie gry i karzą ryzykowne lub nieefektywne zachowanie:

- `-0.01` za każdy krok,
- `-0.2` za próbę wejścia w ścianę lub poza planszę,
- `+1.0` za zebranie skarbu,
- `+5.0` za dotarcie do wyjścia ze skarbem,
- `-0.5` za wejście na wyjście bez skarbu,
- `-2.0` za wejście na pułapkę,
- `-1.0` za przekroczenie limitu kroków.

W treningu Q-learningu stosowane jest dodatkowe kształtowanie nagrody. Jeżeli agent zbliża się do aktualnego celu, czyli do skarbu albo po zebraniu skarbu do wyjścia, dostaje niewielki dodatkowy sygnał treningowy. Wyniki epizodów nadal raportują właściwą nagrodę środowiska.

## Informacje dostępne dla agenta

Obserwacja środowiska jest słownikiem. Zawiera:

- `map_id` - identyfikator aktualnej mapy,
- `agent` - pozycję agenta jako `(row, col)`,
- `treasure` - pozycję skarbu,
- `exit` - pozycję wyjścia,
- `has_treasure` - informację, czy agent zebrał skarb,
- `walls` - macierz ścian,
- `traps` - macierz pułapek.

Środowisko udostępnia więc pełną strukturę planszy. Nie oznacza to jednak, że agent Q-learning używa całej tej informacji jako indeksu stanu. Tablicowy agent RL używa uproszczonego stanu:

```text
(map_id, agent_row, agent_col, has_treasure)
```

Dzięki temu tablica Q pozostaje rozsądnych rozmiarów. Dla aktywnego zestawu map ma wymiary:

```text
liczba_map x wysokość x szerokość x 2 x liczba_akcji
```

Macierze ścian i pułapek są natomiast używane przez baseline BFS, który planuje ścieżkę deterministycznie.

## Budowa agenta Q-learning

Agent Q-learning przechowuje tablicę `Q`, w której dla każdego stanu i każdej akcji zapisana jest przewidywana jakość wykonania tej akcji. Na początku tablica jest wypełniona zerami.

Podczas treningu agent wybiera akcje strategią epsilon-greedy:

- z prawdopodobieństwem `epsilon` wybiera akcję losową, żeby eksplorować środowisko,
- w przeciwnym razie wybiera akcję o największej wartości `Q` dla aktualnego stanu.

W kolejnych epizodach `epsilon` maleje od wartości początkowej do minimalnej. Na początku agent częściej eksperymentuje, a później coraz częściej wykorzystuje wyuczoną politykę.

Aktualizacja tablicy Q ma postać:

```text
Q(s, a) = Q(s, a) + alpha * (r + gamma * max Q(s', a') - Q(s, a))
```

Główne parametry treningu to:

- `alpha` - tempo uczenia,
- `gamma` - dyskonto przyszłych nagród,
- `epsilon_start` - początkowy poziom eksploracji,
- `epsilon_end` - minimalny poziom eksploracji,
- `epsilon_decay` - tempo zmniejszania eksploracji,
- `shaping_scale` - siła dodatkowego sygnału za zbliżanie się do celu.

## Generowanie map

Mapy proceduralne są tworzone w pliku `map_gen.py`. Generator najpierw tworzy labirynt korytarzy algorytmem randomized DFS/backtracking. Parametr `path_width` określa szerokość korytarza w kratkach.

Następnie generator wybiera pozycje `S`, `T` i `E` na podstawie odległości BFS, tak aby start, skarb i wyjście były sensownie rozłożone po mapie. Parametr `difficulty` steruje liczbą pułapek:

- dla wartości `0..1` oznacza gęstość pułapek,
- dla wartości większych niż `1` oznacza dokładną liczbę pułapek.

Pułapki nie są umieszczane na chronionej ścieżce `S -> T -> E`. Po rozmieszczeniu elementów generator dodatkowo sprawdza mapę BFS-em. Jeżeli rozwiązanie nie istnieje, próbuje wygenerować lub rozmieścić elementy ponownie.

## Ocena agenta

Po treningu agent jest oceniany bez eksploracji, czyli z `epsilon = 0`. W każdej turze wybiera najlepszą znaną akcję według tablicy Q. Skrypt raportuje:

- liczbę epizodów,
- skuteczność, czyli procent epizodów zakończonych sukcesem,
- średnią sumę nagród,
- średnią liczbę kroków.

Projekt zawiera też baseline `StrategicBFSAgent`. Ten agent nie uczy się, tylko w każdej turze wyznacza najkrótszą bezpieczną ścieżkę BFS-em:

- przed zebraniem skarbu planuje trasę do `T`,
- po zebraniu skarbu planuje trasę do `E`,
- ignoruje pola będące ścianami i pułapkami.

Baseline BFS służy jako punkt odniesienia. Jeżeli mapa jest rozwiązywalna, BFS powinien osiągać bardzo wysoką skuteczność, ponieważ ma dostęp do pełnej struktury planszy i planuje deterministycznie. Agent Q-learning musi natomiast nauczyć się dobrej polityki przez interakcję ze środowiskiem.
