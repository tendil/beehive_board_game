import copy
import random
import time
from typing import List, Union, Tuple, Dict

import base as Base


class Player(Base.Board):
    def __init__(self, playerName: str, myIsUpper: bool, size: int, myPieces: dict, rivalPieces: dict) -> None:
        """
        Инициализация игрока.

        Args:
        - playerName (str): Имя игрока.
        - myIsUpper (bool): Флаг, указывающий, является ли игрок верхним (True) или нижним (False).
        - size (int): Размер игровой доски.
        - myPieces (dict): Словарь с количеством своих фигур.
        - rivalPieces (dict): Словарь с количеством фигур соперника.
        """
        super().__init__(myIsUpper, size, myPieces, rivalPieces)
        self.playerName: str = playerName
        self.algorithmName: str = "honey"
        self.isUpper: bool = myIsUpper
        self.allies: List[str] = [i for i in self.myPieces]  # type: List[str]
        self.enemies: List[str] = [i for i in self.rivalPieces]  # type: List[str]
        self.directions: List[Tuple[int, int]] = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]  # type: List[Tuple[int, int]]

    def fix_case(self, animal: str) -> str:
        """
        Корректировка регистра фигуры в зависимости от флага isUpper.

        Args:
        - animal (str): Название фигуры.

        Returns:
        - str: Скорректированное название фигуры.
        """
        return animal.upper() if self.isUpper else animal.lower()

    def get_all_empty_cells(self) -> List[List[int]]:
        """
        Получение координат всех пустых клеток на доске.

        Returns:
        - List[List[int]]: Список координат пустых клеток.
        """
        return [[p, q] for p in self.board for q in self.board[p] if self.isEmpty(p, q, self.board)]

    def get_all_nonempty_cells(self) -> List[List[int]]:
        """
        Получение координат всех непустых клеток на доске.

        Returns:
        - List[List[int]]: Список координат непустых клеток.
        """
        return [[p, q] for p in self.board for q in self.board[p] if not self.isEmpty(p, q, self.board)]

    def get_all_my_cells(self) -> List[List[int]]:
        """
        Получение координат всех клеток с фигурами текущего игрока.

        Returns:
        - List[List[int]]: Список координат клеток с фигурами текущего игрока.
        """
        return [[p, q] for p in self.board for q in self.board[p] if not self.isEmpty(p, q, self.board) and self.board[p][q][-1].isupper() == self.isUpper]

    def make_move(self) -> List:
        """
        Определение следующего хода игрока.

        Returns:
        - List: Список, представляющий следующий ход игрока.
        """
        if self.should_start():
            return self.starting_move()

        if self.myMove == 1:
            return self.place_piece(self.fix_case("a"))

        if self.myMove == 2:
            return self.place_piece(self.fix_case("q"))

        return self.random_move()

    def should_start(self) -> bool:
        """
        Проверка, должен ли игрок начать игру.

        Returns:
        - bool: True, если игрок должен начать игру, False в противном случае.
        """
        return sum(1 for p in self.board for q in self.board[p] if not self.isEmpty(p, q, self.board)) < 2

    def starting_move(self) -> List:
        """
        Определение начального хода игрока.

        Returns:
        - List: Список, представляющий начальный ход игрока.
        """
        if self.count_nonempty_cells() == 0:
            return [self.fix_case("g"), None, None, 3, 6]
        elif self.count_nonempty_cells() == 1:
            p, q = random.choice(self.get_all_nonempty_cells())
            rand_direction = random.choice(self.directions)
            return [self.fix_case("g"), None, None, p + rand_direction[0], q + rand_direction[1]]

    def place_piece(self, piece_type: str) -> List:
        """
        Размещение фигуры на доске.

        Args:
        - piece_type (str): Тип фигуры.

        Returns:
        - List: Список, представляющий размещение фигуры на доске.
        """
        p, q = self.randomly_place()
        if self.myPieces[self.fix_case(piece_type)] == 0:
            return [self.get_random_piece(), None, None, p, q]
        return [self.fix_case(piece_type), None, None, p, q]

    def random_move(self) -> List:
        """
        Генерация случайного хода.

        Returns:
        - List: Список, представляющий случайный ход.
        """
        random_move = self.random()
        if random_move is not None:
            animal, p, q, new_p, new_q = random_move
            return [animal, p, q, new_p, new_q]
        return []

    def count_nonempty_cells(self) -> int:
        """
        Подсчет количества непустых клеток на доске.

        Returns:
        - int: Количество непустых клеток.
        """
        return sum(1 for p in self.board for q in self.board[p] if not self.isEmpty(p, q, self.board))

    def get_random_piece(self) -> str:
        """
        Получение случайной фигуры из доступных.

        Returns:
        - str: Название случайной фигуры.
        """
        available_animals = [animal for animal, count in self.myPieces.items() if count > 0]
        if not available_animals:
            return False
        return random.choice(available_animals)

    def valid_placement(self, p: int, q: int) -> bool:
        """
        Проверка возможности размещения фигуры на указанных координатах.

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - bool: True, если размещение возможно, False в противном случае.
        """
        if not self.isEmpty(p, q, self.board) or not self.inBoard(p, q):
            return False

        num_of_allies = sum(1 for i in range(6) if
                            self.inBoard(p + self.directions[i][0], q + self.directions[i][1]) and
                            self.board[p + self.directions[i][0]][q + self.directions[i][1]] in self.allies)

        if num_of_allies > 0:
            return True
        return False

    def get_valid_moves(self, p: int, q: int) -> List:
        """
        Получение списка допустимых ходов для фигуры на указанных координатах.

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - List: Список допустимых ходов.
        """
        animal = self.board[p][q][-1].lower()
        moves: List = []

        if animal == "a":
            moves = self.get_valid_moves_ant(p, q)
        elif animal == "b":
            moves = self.get_valid_moves_beetle(p, q)
        elif animal == "g":
            moves = self.get_valid_moves_grasshopper(p, q)
        elif animal == "q":
            moves = self.get_valid_moves_queen(p, q)
        elif animal == "s":
            moves = self.get_valid_moves_spider(p, q)

        return moves

    def randomly_place(self) -> Tuple[int, int]:
        """
        Выбор случайной позиции для размещения фигуры.

        Returns:
        - Tuple[int, int]: Кортеж с координатами выбранной позиции.
        """
        placements = [[p, q] for p in self.board for q in self.board[p] if self.valid_placement(p, q)]
        if not placements:
            return None, None
        return random.choice(placements)

    def surroundings(self, p: int, q: int) -> List[Union[int, List[int]]]:
        """
        Получение координат окружающих клеток.

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - List[Union[int, List[int]]]: Список координат окружающих клеток.
        """
        o = [0] * 6
        for i in range(6):
            pos = [p + self.directions[i][0], q + self.directions[i][1]]
            o[i] = pos if self.inBoard(pos[0], pos[1]) else -1
        return o

    def neighbours(self, p: int, q: int) -> List[List[int]]:
        """
        Получение координат соседних клеток.

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - List[List[int]]: Список координат соседних клеток.
        """
        surroundings = self.surroundings(p, q)
        return [[k[0], k[1]] for k in surroundings if
                k != -1 and self.board[k[0]][k[1]] and self.board[k[0]][k[1]][-1] in self.enemies + self.allies]

    def find_island(self) -> List[List[int]]:
        """
        Поиск "острова" - группы смежных клеток с фигурами.

        Returns:
        - List[List[int]]: Список координат клеток "острова".
        """
        island: List[List[int]] = []

        for p in self.board:
            for q in self.board[p]:
                if self.board[p][q] != "":
                    island.append([p, q])

        i = 0
        while i < len(island):
            cell = island[i]
            neighbours = self.neighbours(cell[0], cell[1])
            for k in neighbours:
                if k not in island:
                    island.append(k)
            i += 1

        return island

    def will_break_hive(self, p: int, q: int, newP: int, newQ: int) -> bool:
        """
        Проверка, разорвет ли ход улей (группу фигур) на доске.

        Args:
        - p (int): Исходная координата p.
        - q (int): Исходная координата q.
        - newP (int): Новая координата p.
        - newQ (int): Новая координата q.

        Returns:
        - bool: True, если ход разорвет улей, False в противном случае.
        """
        test_board = copy.deepcopy(self.board)

        animal = test_board[p][q][-1]
        test_board[p][q] = test_board[p][q][:-1]

        original_island = self.find_island()
        new_island = self.find_island()

        if set(map(tuple, original_island)) != set(map(tuple, new_island)):
            return True

        if test_board[newP][newQ] != "":
            return False

        test_board[newP][newQ] += animal

        new_island = self.find_island()

        if set(map(tuple, original_island)) != set(map(tuple, new_island)):
            return True

        return False

    def check_moves(self, p: int, q: int, moves: List[List[int]]) -> List[List[int]]:
        """
        Проверка и отбор ходов, которые не разрушат улей.

        Args:
        - p (int): Координата p.
        - q (int): Координата q.
        - moves (List[List[int]]): Список возможных ходов.

        Returns:
        - List[List[int]]: Отфильтрованный список ходов.
        """
        checked_moves = [i for i in moves if self.inBoard(i[0], i[1]) and not self.will_break_hive(p, q, i[0], i[1])]
        return checked_moves

    def asq_move(self, p: int, q: int, board: List[List[str]]) -> List[List[int]]:
        """
        Получение возможных ходов для фигуры типа "муравей".

        Args:
        - p (int): Координата p.
        - q (int): Координата q.
        - board (List[List[str]]): Игровая доска.

        Returns:
        - List[List[int]]: Список возможных ходов.
        """
        o = [0] * 6

        for i in range(6):
            if self.inBoard(p + self.directions[i][0], q + self.directions[i][1]):
                pos = [p + self.directions[i][0], q + self.directions[i][1]]
                if pos[1] in board[pos[0]]:
                    pos_content = board[pos[0]][pos[1]]
                    if pos_content:
                        pos_content = pos_content[-1]
                        if pos_content in self.enemies + self.allies:
                            o[i] = 1
                        else:
                            o[i] = 0
                else:
                    o[i] = -1

        possible_moves = [
            [p + self.directions[i][0], q + self.directions[i][1]]
            for i in range(6)
            if o[i] == 0 and (
                    (o[(i - 1) % 6] == 0 and o[(i + 1) % 6] == 1) or
                    (o[(i - 1) % 6] == 1 and o[(i + 1) % 6] == 0))
        ]

        return possible_moves

    def pos_pos(self, p: int, q: int, n: int) -> List[List[int]]:
        """
        Получение позиций на доске, достижимых за n ходов.

        Args:
        - p (int): Исходная координата p.
        - q (int): Исходная координата q.
        - n (int): Количество ходов.

        Returns:
        - List[List[int]]: Список достижимых позиций.
        """
        board = copy.deepcopy(self.board)
        o_p, o_q = p, q

        pos = [[p, q]]
        dont_go_back = pos[:]
        board[p][q] = ""

        for _ in range(n):
            pos2 = [x for possible in pos for x in self.asq_move(possible[0], possible[1], board) if
                    x not in dont_go_back]
            dont_go_back.extend(pos2)

        posPos = self.check_moves(o_p, o_q, pos2)
        return posPos


    def get_valid_moves_spider(self, p: int, q: int) -> List[List[int]]:
        """
        Получение всех возможных ходов для фигуры "паук".

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - List[List[int]]: Список возможных ходов.
        """
        return self.pos_pos(p, q, 3)

    def get_valid_moves_queen(self, p: int, q: int) -> List[List[int]]:
        """
        Получение всех возможных ходов для фигуры "королева".

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - List[List[int]]: Список возможных ходов.
        """
        return self.pos_pos(p, q, 1)

    def get_valid_moves_ant(self, p: int, q: int) -> List[List[int]]:
        """
        Получение всех возможных ходов для фигуры "муравей".

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - List[List[int]]: Список возможных ходов.
        """
        moves: List[List[int]] = []
        n = 1
        while True:
            addition = self.pos_pos(p, q, n)

            if not addition:
                break

            moves.extend([j for j in addition if j not in moves])
            n += 1

        return moves

    def get_valid_moves_beetle(self, p: int, q: int) -> List[List[int]]:
        """
        Получение всех возможных ходов для фигуры "жук".

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - List[List[int]]: Список возможных ходов.
        """
        surroundings = self.surroundings(p, q)
        moves = [i for i in surroundings if i != -1]
        moves = self.check_moves(p, q, moves)
        return moves

    def get_valid_moves_grasshopper(self, p: int, q: int) -> List[List[int]]:
        """
        Получение всех возможных ходов для фигуры "кузнечик".

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - List[List[int]]: Список возможных ходов.
        """
        moves: List[List[int]] = []
        for i in range(6):
            new_p, new_q = p, q
            p_dir = p + self.directions[i][0]
            q_dir = q + self.directions[i][1]
            if self.inBoard(p_dir, q_dir):
                while True:
                    new_p += self.directions[i][0]
                    new_q += self.directions[i][1]
                    if not self.inBoard(new_p, new_q):
                        break
                    if self.board[new_p][new_q] == "":
                        moves.append([new_p, new_q])
                        break

        moves = self.check_moves(p, q, moves)
        return moves

    def find_queen(self) -> Tuple[int, int]:
        """
        Поиск координат королевы на доске.

        Returns:
        - Tuple[int, int]: Кортеж с координатами королевы.
        """
        animal = "q"
        if self.isUpper:
            animal = animal.lower()
        else:
            animal = animal.upper()

        for p in self.board:
            for q in self.board[p]:
                if animal in self.board[p][q]:
                    return p, q

        return None, None

    def value_tiles(self, p: int, q: int, depth: int = 0, values: Dict[int, Dict[int, int]] = None, direction: int = 0,
                    org: bool = True) -> Dict[int, Dict[int, int]]:
        """
        Оценка значимости каждой клетки на доске.

        Args:
        - p (int): Координата p.
        - q (int): Координата q.
        - depth (int): Глубина рекурсии.
        - values (Dict[int, Dict[int, int]]): Словарь значений клеток.
        - direction (int): Направление рекурсии.
        - org (bool): Флаг, указывающий, является ли вызов началом оценки.

        Returns:
        - Dict[int, Dict[int, int]]: Словарь значений клеток.
        """
        if values is None:
            values = copy.deepcopy(self.board)
            for i in values:
                for j in values[i]:
                    values[i][j] = -1

            values[p][q] = 0
            for i in range(6):
                p1 = p + self.directions[i][0]
                q1 = q + self.directions[i][1]
                self.value_tiles(p1, q1, depth + 1, values, i, True)

            return values

        if not self.inBoard(p, q):
            return

        values[p][q] = depth

        dir1 = self.directions[direction % 6]
        dir2 = self.directions[(direction + 1) % 6]
        p1 = p + dir1[0]
        q1 = q + dir1[1]

        p2 = p + dir2[0]
        q2 = q + dir2[1]

        self.value_tiles(p1, q1, depth + 1, values, direction, org)

        if org:
            self.value_tiles(p2, q2, depth + 1, values, direction + 1, False)

    def devalue_around_queen(self, values: Dict[int, Dict[int, int]]) -> Dict[int, Dict[int, int]]:
        """
        Уменьшение значений клеток вокруг королевы.

        Args:
        - values (Dict[int, Dict[int, int]]): Словарь значений клеток.

        Returns:
        - Dict[int, Dict[int, int]]: Словарь значений клеток.
        """
        animal = "q"
        if self.isUpper:
            animal = animal.upper()
        else:
            animal = animal.lower()

        queen_P, queen_Q = next(((p, q) for p, row in self.board.items() for q, cell in row.items() if animal in cell),
                                (None, None))

        if queen_P is None or queen_Q is None:
            return values

        num_of_neighbours = sum(
            1 for i in range(6) if self.inBoard(queen_P + self.directions[i][0], queen_Q + self.directions[i][1]))

        for i in range(6):
            p1 = queen_P + self.directions[i][0]
            q1 = queen_Q + self.directions[i][1]
            if self.inBoard(p1, q1):
                values[p1][q1] += num_of_neighbours + 1

        return values

    def assign_value(self, animal: str) -> int:
        """
        Присвоение значения фигуре в зависимости от её типа.

        Args:
        - animal (str): Тип фигуры.

        Returns:
        - int: Значение фигуры.
        """
        animal = animal.lower()
        values = {"a": 3, "b": 8, "s": 7, "g": 5}
        return values.get(animal, 0)

    def get_best_move(self, moves: Dict[Union[str, Tuple[int, int]], List[List[int]]]) -> List[Union[str, int]]:
        """
        Выбор лучшего хода с учётом оценки значимости клеток.

        Args:
        - moves (Dict[Union[str, Tuple[int, int]], List[List[int]]]): Словарь ходов.

        Returns:
        - List[Union[str, int]]: Лучший ход.
        """
        best_move: List[Union[str, int]] = []
        pQ, qQ = self.find_queen()
        if pQ is None or qQ is None:
            return best_move
        tile_values = self.value_tiles(pQ, qQ)
        tile_values = self.devalue_around_queen(tile_values)
        best_value = -1
        for i, moves_list in moves.items():
            if type(i) is str:
                p, q = None, None
            else:
                p, q = i

            if p is not None:
                value = tile_values[p][q]
                if self.board[p][q][-1] in ["Q", "q"]:
                    value += 10
                if 1 <= value <= 2:
                    continue
            else:
                value = self.assign_value(i)

            for j in moves_list:
                newP, newQ = j
                new_value = tile_values[newP][newQ]
                delta = value - new_value
                abs_delta = abs(delta)
                if abs_delta > best_value:
                    best_value = abs_delta
                    best_move = [p, q, newP, newQ, i]

        return best_move

    def all_placements(self) -> Dict[str, List[List[int]]]:
        """
        Получение всех возможных размещений фигур.

        Returns:
        - Dict[str, List[List[int]]]: Словарь размещений фигур.
        """
        placements: Dict[str, List[List[int]]] = {}

        for i in self.myPieces:
            if self.myPieces[i] > 0:
                placements[i] = [[p, q] for p, q in self.board.items() for q in self.board[p] if
                                 self.valid_placement(p, q)]

        return placements

    def random_available_animal(self) -> Union[str, bool]:
        """
        Выбор случайной доступной фигуры.

        Returns:
        - Union[str, bool]: Случайная фигура или False, если нет доступных фигур.
        """
        available_animals = [i for i in self.myPieces if self.myPieces[i] > 0]
        return random.choice(available_animals) if available_animals else False

    def random(self) -> List[Union[str, int]]:
        """
        Случайный ход.

        Returns:
        - List[Union[str, int]]: Случайный ход.
        """
        placements = self.all_placements()
        all_my_cells = self.get_all_my_cells()
        moves = {(p, q): self.get_valid_moves(p, q) for p, q in all_my_cells}
        all_moves = placements.copy()
        all_moves.update(moves)

        if self.myMove == 3:
            animal = self.get_random_piece()
            p, q = self.randomly_place()
            return [animal, None, None, p, q]

        best_move = self.get_best_move(all_moves)
        if best_move:
            if best_move[0] is None and best_move[1] is None:
                return [best_move[4], None, None, best_move[2], best_move[3]]
            return [self.board[best_move[0]][best_move[1]][-1], best_move[0], best_move[1], best_move[2], best_move[3]]

        for (p, q) in moves:
            if moves[(p, q)]:
                break

        if not moves[(p, q)]:
            return None

        new_p, new_q = random.choice(moves[(p, q)])
        animal = self.board[p][q][-1]
        return [animal, p, q, new_p, new_q]

    def is_queen_surrounded(self, p: int, q: int) -> bool:
        """
        Проверка окружения королевы.

        Args:
        - p (int): Координата p.
        - q (int): Координата q.

        Returns:
        - bool: True, если королева окружена, иначе False.
        """
        surroundings = self.surroundings(p, q)
        for pos in surroundings:
            if pos != -1 and self.isEmpty(pos[0], pos[1], self.board):
                return False
        return True


def check_game_over(player1, player2):
    """Проверка завершения игры на основе условия с королевами."""
    queen1_P, queen1_Q = player1.find_queen()
    queen2_P, queen2_Q = player2.find_queen()

    if queen1_P is None or queen1_Q is None or queen2_P is None or queen2_Q is None:
        return False

    if player1.is_queen_surrounded(queen1_P, queen1_Q) or player2.is_queen_surrounded(queen2_P, queen2_Q):
        return True

    return False


def update_players(move, activePlayer, passivePlayer):
    """Обновление состояния игроков на основе хода."""
    if not move:
        return

    animal, p, q, new_p, new_q = move

    if p is None and q is None:
        activePlayer.myPieces[animal] -= 1
        passivePlayer.rivalPieces = activePlayer.myPieces.copy()
        print("P1 returned", [animal, p, q, new_p, new_q])
    else:
        activePlayer.board[p][q] = activePlayer.board[p][q][:-1]
        passivePlayer.board[p][q] = passivePlayer.board[p][q][:-1]
        print("P2 returned", [animal, p, q, new_p, new_q])

    activePlayer.board[new_p][new_q] += animal
    passivePlayer.board[new_p][new_q] += animal


def main():
    small_figures = {
        "q": 1,
        "a": 2,
        "b": 2,
        "s": 2,
        "g": 2,
    }

    big_figures = {figure.upper(): small_figures[figure] for figure in small_figures}

    player1 = Player("player1", False, 13, small_figures, big_figures)
    player2 = Player("player2", True, 13, big_figures, small_figures)

    move_idx = 0
    while True:
        move = player1.make_move()
        update_players(move, player1, player2)
        player1.saveImage("img_moves/move-{:03d}-player1.png".format(move_idx))
        time.sleep(0.5)

        if check_game_over(player1, player2):
            print("Player 1 won!")
            break

        move = player2.make_move()
        update_players(move, player2, player1)
        player1.saveImage("img_moves/move-{:03d}-player2.png".format(move_idx))
        time.sleep(0.5)

        if check_game_over(player1, player2):
            print("Player 2 won!")
            break

        move_idx += 1
        player1.myMove = move_idx
        player2.myMove = move_idx

        if move_idx > 50:
            print("End of the game")
            break


if __name__ == "__main__":
    main()
