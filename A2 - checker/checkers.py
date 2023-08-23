from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

# debug constants
DEBUG = False
PROFILE = False
LOGGING = True
TESTING = False
RUN_ALL_TESTS = False
USE_EVAL_ORDERING = False
USE_BINARY_BOARD = False
USE_EVAL2 = False

# board constants
BOARD_WIDTH, BOARD_HEIGHT = 8, 8
EMPTY = '.'
RED = 'r'
RED_KING = 'R'
BLACK = 'b'
BLACK_KING = 'B'
RED_TEAM = [RED, RED_KING]
BLACK_TEAM = [BLACK, BLACK_KING]
OPPONENTS = {
    RED: BLACK_TEAM,
    RED_KING: BLACK_TEAM,
    BLACK: RED_TEAM,
    BLACK_KING: RED_TEAM
}

# move constants
JUMP_DIRECTIONS = {
    RED: [(1, -1), (-1, -1)],
    RED_KING: [(1, 1), (-1, -1), (1, -1), (-1, 1)],
    BLACK: [(1, 1), (-1, 1)],
    BLACK_KING: [(1, 1), (-1, -1), (1, -1), (-1, 1)]
}

# player constants
RED_PLAYER = 1
BLACK_PLAYER = -1
PLAYER_TO_PIECES = {
    RED_PLAYER: RED_TEAM,
    BLACK_PLAYER: BLACK_TEAM
}

# search constants
DEPTH_LIMIT = 14
INFINITY = sys.maxsize // 100

# evaluation constants
CENTER_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

# global variables
#                     depth, type,       utility, move
cache: Dict[State, List[int, int, List[int, Optional[Board]]]] = {}
# note that the type is in {-1, 0, 1}
# where -1 denotes lowerbound, 0 denotes exact, and 1 denotes upperbound
LOWER_BOUND = -1
EXACT = 0
UPPER_BOUND = 1

eval_cache: Dict[State, int] = {}


class GridBoard:
    grid: List[List[str]]
    pieces_count: Dict[str, int]
    is_finalized: bool
    hash_value: int

    def __init__(self, board: GridBoard = None, input_file: str = None):
        self.is_finalized = False
        if input_file is None and board is None:
            raise ValueError('Must provide either input_file or board')
        elif input_file is not None:
            self.pieces_count = defaultdict(int)
            self._read_board(input_file)
        elif board is not None:
            # self.grid = deepcopy(board.grid)
            self.grid = [row[:] for row in board.grid]
            # copy a piece count
            self.pieces_count = defaultdict(int, board.pieces_count)

    def get_count_difference(self) -> int:
        """
        :return: returns ((red count, red king count), (black count, black king count))
        """
        red, red_king, black, black_king = self.pieces_count[RED], self.pieces_count[RED_KING], \
            self.pieces_count[BLACK], self.pieces_count[BLACK_KING]
        return (red + red_king << 2) - (black + black_king << 2)

    def get_winner(self) -> int:
        """
        :return: the player who win the game, or 0 if no one winning
        """
        reds = self.pieces_count[RED] + self.pieces_count[RED_KING]
        blacks = self.pieces_count[BLACK] + self.pieces_count[BLACK_KING]
        if reds != 0 and blacks == 0:
            return RED_PLAYER
        elif reds == 0 and blacks != 0:
            return BLACK_PLAYER
        else:
            return 0

    def get_moves(self, player: int) -> List[GridBoard]:
        ans = []
        # get any jumps
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.get(x, y) in PLAYER_TO_PIECES[player]:
                    jumps = self.get_jumps(x, y)
                    ans.extend(jumps)
        # must jump
        if ans:
            return ans
        # no jumps, get simple moves
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.get(x, y)
                if piece not in PLAYER_TO_PIECES[player]:
                    continue
                for dx, dy in JUMP_DIRECTIONS[piece]:
                    nx, ny = x + dx, y + dy
                    if not self.is_valid(nx, ny):
                        continue
                    if self.get(nx, ny) != EMPTY:
                        continue
                    new_board = GridBoard(self)
                    new_board.set(x, y, EMPTY)
                    if piece == RED and ny == 0:
                        new_board.set(nx, ny, RED_KING)
                    elif piece == BLACK and ny == BOARD_HEIGHT - 1:
                        new_board.set(nx, ny, BLACK_KING)
                    else:
                        new_board.set(nx, ny, piece)
                    new_board.set_finalized()
                    ans.append(new_board)
        return ans

    def get_jumps(self, x: int, y: int) -> List[GridBoard]:
        # base case: no jump possible -> return []
        res = []
        piece = self.get(x, y)
        for dx, dy in JUMP_DIRECTIONS[piece]:
            ox, oy = x + dx, y + dy
            if not self.is_valid(ox, oy) or self.get(ox, oy) not in OPPONENTS[piece]:
                continue
            nx, ny = ox + dx, oy + dy
            if not self.is_valid(nx, ny) or self.get(nx, ny) != EMPTY:
                continue
            # jump and capture
            new_board = GridBoard(self)
            new_board.set(x, y, EMPTY)
            new_board.set(ox, oy, EMPTY)
            # king case (if become king, stop jumping)
            if piece == RED and ny == 0:
                new_board.set(nx, ny, RED_KING)
            elif piece == BLACK and ny == BOARD_HEIGHT - 1:
                new_board.set(nx, ny, BLACK_KING)
            else:
                # multiple jump case
                new_board.set(nx, ny, piece)
                # if we can jump to some other positions in new board
                new_jumps = new_board.get_jumps(nx, ny)
                if new_jumps:
                    res.extend(new_jumps)
                    continue
            new_board.set_finalized()
            res.append(new_board)
        return res

    def set_finalized(self):
        self.hash_value = self.__hash__()
        self.is_finalized = True

    def get(self, x: int, y: int):
        return self.grid[y][x]

    def __getitem__(self, item):
        x, y = item
        return self.grid[y][x]

    def set(self, x: int, y: int, val: str):
        assert not self.is_finalized
        self.pieces_count[self.grid[y][x]] -= 1
        self.pieces_count[val] += 1
        self.grid[y][x] = val
        # if DEBUG:
        #     red, red_king, black, black_king = 0, 0, 0, 0
        #     for y in range(BOARD_HEIGHT):
        #         for x in range(BOARD_WIDTH):
        #             p = self.get(x, y)
        #             if p == '.':
        #                 continue
        #             if p == RED:
        #                 red += 1
        #             elif p == RED_KING:
        #                 red_king += 1
        #             elif p == BLACK:
        #                 black += 1
        #             elif p == BLACK_KING:
        #                 black_king += 1
        #     if any((red != self.pieces_count[RED], red_king != self.pieces_count[RED_KING],
        #             black != self.pieces_count[BLACK],
        #             black_king != self.pieces_count[BLACK_KING])):
        #         print(self)
        #         print(red, self.pieces_count[RED])
        #         print(red_king, self.pieces_count[RED_KING])
        #         print(black, self.pieces_count[BLACK])
        #         print(black_king, self.pieces_count[BLACK_KING])
        #         print(self.pieces_count)
        #         raise Exception('Invalid board state')

    def __setitem__(self, key, value):
        x, y = key
        self.set(x, y, value)

    @staticmethod
    def is_valid(x: int, y: int):
        """Precondition: x and y are offset only by 1 from valid coordinates."""
        return not (x < 0 or y < 0 or x >= BOARD_WIDTH or y >= BOARD_HEIGHT)

    def _read_board(self, input_file: str) -> None:
        self.grid = []
        with open(input_file) as file:
            lines = file.readlines()
            for y in range(BOARD_HEIGHT):
                self.grid.append([])
                for x in range(BOARD_WIDTH):
                    p = lines[y][x]
                    self.grid[y].append(p)
                    self.pieces_count[p] += 1
        self.set_finalized()

    def __str__(self) -> str:
        s = ''
        for line in self.grid:
            s += ''.join(line) + '\n'
        return s

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        if self.is_finalized:
            return self.hash_value
        return hash(self.__str__())

    def __eq__(self, other):
        # if not isinstance(other, self.__class__):
        #     return False
        return self.grid == other.grid

    def __lt__(self, other):
        # if not isinstance(other, self.__class__):
        #     return False
        return self.get_count_difference() < other.get_count_difference()


class BinaryBoard:
    red_men: int
    red_kings: int
    black_men: int
    black_kings: int

    pieces_count: defaultdict[str, int]

    def __init__(self, board: BinaryBoard = None, input_file: str = None):
        self.red_men = 0
        self.red_kings = 0
        self.black_men = 0
        self.black_kings = 0
        if input_file is not None:
            self.pieces_count = defaultdict(int)
            self._read_board(input_file)
        elif board is not None:
            self.red_men = board.red_men
            self.red_kings = board.red_kings
            self.black_men = board.black_men
            self.black_kings = board.black_kings
            self.pieces_count = defaultdict(int, board.pieces_count)
        else:
            raise ValueError

    def get_moves(self, player: int) -> List[BinaryBoard]:
        ans = []
        # get any jumps
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self[x, y] in PLAYER_TO_PIECES[player]:
                    jumps = self.get_jumps(x, y)
                    ans.extend(jumps)
        # must jump
        if ans:
            return ans
        # no jumps, get simple moves
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self[x, y]
                if piece not in PLAYER_TO_PIECES[player]:
                    continue
                for dx, dy in JUMP_DIRECTIONS[piece]:
                    nx, ny = x + dx, y + dy
                    if not self.is_valid(nx, ny):
                        continue
                    if self[nx, ny] != EMPTY:
                        continue
                    new_board = BinaryBoard(self)
                    new_board[x, y] = EMPTY
                    if piece == RED and ny == 0:
                        new_board[nx, ny] = RED_KING
                    elif piece == BLACK and ny == BOARD_HEIGHT - 1:
                        new_board[nx, ny] = BLACK_KING
                    else:
                        new_board[nx, ny] = piece
                    ans.append(new_board)
        return ans

    def get_jumps(self, x: int, y: int) -> List[BinaryBoard]:
        res = []
        piece = self[x, y]
        for dx, dy in JUMP_DIRECTIONS[piece]:
            ox, oy = x + dx, y + dy
            if not self.is_valid(ox, oy) or self[ox, oy] not in OPPONENTS[piece]:
                continue
            nx, ny = ox + dx, oy + dy
            if not self.is_valid(nx, ny) or self[nx, ny] != EMPTY:
                continue
            new_board = BinaryBoard(self)
            new_board[x, y] = EMPTY
            new_board[ox, oy] = EMPTY
            if piece == RED and ny == 0:
                new_board[nx, ny] = RED_KING
            elif piece == BLACK and ny == BOARD_HEIGHT - 1:
                new_board[nx, ny] = BLACK_KING
            else:
                # multiple jump case
                new_board[nx, ny] = piece
                # if we can jump to some other positions in new board
                new_jumps = new_board.get_jumps(nx, ny)
                if new_jumps:
                    res.extend(new_jumps)
                    continue
            res.append(new_board)
        return res

    def __getitem__(self, item) -> str:
        x, y = item
        offset = y * BOARD_WIDTH + x
        if 1 & self.red_men >> offset:
            return RED
        elif 1 & self.red_kings >> offset:
            return RED_KING
        elif 1 & self.black_men >> offset:
            return BLACK
        elif 1 & self.black_kings >> offset:
            return BLACK_KING
        else:
            return EMPTY

    def __setitem__(self, key, value):
        x, y = key
        offset = y * BOARD_WIDTH + x
        v = 1 << offset
        if value == RED:
            self.red_men |= v
            self.pieces_count[RED] += 1
        elif value == RED_KING:
            self.red_kings |= v
            self.pieces_count[RED_KING] += 1
        elif value == BLACK:
            self.black_men |= v
            self.pieces_count[BLACK] += 1
        elif value == BLACK_KING:
            self.black_kings |= v
            self.pieces_count[BLACK_KING] += 1
        else:
            self.pieces_count[self[x, y]] -= 1
            v = ~v
            self.red_men &= v
            self.red_kings &= v
            self.black_men &= v
            self.black_kings &= v

    @staticmethod
    def is_valid(x: int, y: int):
        """Precondition: x and y are offset only by 1 from valid coordinates."""
        return not (x < 0 or y < 0 or x >= BOARD_WIDTH or y >= BOARD_HEIGHT)

    def _read_board(self, input_file: str):
        with open(input_file) as file:
            lines = file.readlines()
            for y in range(BOARD_HEIGHT - 1, -1, -1):
                for x in range(BOARD_WIDTH - 1, -1, -1):
                    self.red_men <<= 1
                    self.red_kings <<= 1
                    self.black_men <<= 1
                    self.black_kings <<= 1
                    p = lines[y][x]
                    if p == 'r':
                        self.red_men |= 1
                    elif p == 'R':
                        self.red_kings |= 1
                    elif p == 'b':
                        self.black_men |= 1
                    elif p == 'B':
                        self.black_kings |= 1
                    self.pieces_count[p] += 1

    def get_winner(self) -> int:
        """
        :return: the player who win the game, or 0 if no one winning
        """
        reds = self.pieces_count[RED] + self.pieces_count[RED_KING]
        blacks = self.pieces_count[BLACK] + self.pieces_count[BLACK_KING]
        if reds != 0 and blacks == 0:
            return RED_PLAYER
        elif reds == 0 and blacks != 0:
            return BLACK_PLAYER
        else:
            return 0

    def get_count_difference(self) -> int:
        """
        :return: returns ((red count, red king count), (black count, black king count))
        """
        red, red_king, black, black_king = self.pieces_count[RED], self.pieces_count[RED_KING], \
            self.pieces_count[BLACK], self.pieces_count[BLACK_KING]
        return (red + red_king << 2) - (black + black_king << 2)

    def __repr__(self):
        return f'red_men: {bin(self.red_men)}\nred_kings: {bin(self.red_kings)}\n' \
               f'black_men: {bin(self.black_men)}\nblack_kings: {bin(self.black_kings)}\n' + \
            self.get_grid_str()

    def get_grid_str(self) -> str:
        grid = [['' for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                grid[y][x] = self[x, y]
        s = ''
        for line in grid:
            s += ''.join(line) + '\n'
        return s

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.red_men == other.red_men and self.red_kings == other.red_kings and \
            self.black_men == other.black_men and self.black_kings == other.black_kings

    def __hash__(self):
        return hash((self.red_men, self.red_kings, self.black_men, self.black_kings))

    def __lt__(self, other):
        return self.get_count_difference() < other.get_count_difference()


Board = BinaryBoard if USE_BINARY_BOARD else GridBoard


def evaluate2(self) -> int:
    if self in eval_cache:
        return eval_cache[self]
    red_pawns, red_kings, black_pawns, black_kings = self.board.pieces_count[RED], \
        self.board.pieces_count[RED_KING], self.board.pieces_count[BLACK], \
        self.board.pieces_count[BLACK_KING]
    red_safe_pawns, red_safe_kings, black_safe_pawns, black_safe_kings = 0, 0, 0, 0
    red_attack_pawns, red_attack_kings, black_attack_pawns, black_attack_kings = 0, 0, 0, 0
    red_center_pawns, red_center_kings, black_center_pawns, black_center_kings = 0, 0, 0, 0
    red_defend_pawns, red_defend_kings, black_defend_pawns, black_defend_kings = 0, 0, 0, 0
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            p = self.board[x, y]
            if p == EMPTY:
                pass
            elif p == RED:
                if y == 0 or y == BOARD_HEIGHT - 1 or x == 0 or x == BOARD_WIDTH - 1:
                    red_safe_pawns += 1
                if y < 3:
                    red_attack_pawns += 1
                if y > 4:
                    red_defend_pawns += 1
                red_center_pawns += CENTER_MAP[y][x]
            elif p == RED_KING:
                if y == 0 or y == BOARD_HEIGHT - 1 or x == 0 or x == BOARD_WIDTH - 1:
                    red_safe_kings += 1
                if y < 3:
                    red_attack_kings += 1
                if y > 4:
                    red_defend_kings += 1
                red_center_kings += CENTER_MAP[y][x]
            elif p == BLACK:
                if y == 0 or y == BOARD_HEIGHT - 1 or x == 0 or x == BOARD_WIDTH - 1:
                    black_safe_pawns += 1
                if y > 4:
                    black_attack_pawns += 1
                if y < 3:
                    black_defend_pawns += 1
                black_center_pawns += CENTER_MAP[y][x]
            elif p == BLACK_KING:
                if y == 0 or y == BOARD_HEIGHT - 1 or x == 0 or x == BOARD_WIDTH - 1:
                    black_safe_kings += 1
                if y > 4:
                    black_attack_kings += 1
                if y < 3:
                    black_defend_kings += 1
                black_center_kings += CENTER_MAP[y][x]
    ans = 100 * (red_pawns + (red_kings << 1) - black_pawns - (black_kings << 1)) \
          + (red_safe_pawns + red_safe_kings - black_safe_pawns - black_safe_kings) + \
          50 * (red_attack_pawns + red_attack_kings -
                black_attack_pawns - black_attack_kings) + \
          50 * (red_center_pawns + red_center_kings -
                black_center_pawns - black_center_kings) + \
          10 * (red_defend_pawns + red_defend_kings - black_defend_pawns - black_defend_kings)
    eval_cache[self] = ans
    return ans


def evaluate1(self) -> int:
    """Precondition: self is not a terminal state."""
    if self in eval_cache:
        return eval_cache[self]
    # piece count
    difference = self.board.get_count_difference()

    # center control
    red_center, black_center = 0, 0
    # become king
    red_to_king_distance, black_to_king_distance = 0, 0
    # defensive structure
    defensive_directions = ((2, 0), (-2, 0), (0, 2), (0, -2))
    red_defensive_count, black_defensive_count = 0, 0
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            p = self.board[x, y]
            if p == EMPTY:
                continue
            if p == RED:
                if CENTER_MAP[y][x] == 1:
                    red_center += 1
                if difference < 0:
                    red_to_king_distance += y
            elif p == RED_KING:
                if CENTER_MAP[y][x] == 1:
                    red_center += 10
            elif p == BLACK:
                if CENTER_MAP[y][x] == 1:
                    black_center += 1
                if difference > 0:
                    black_to_king_distance += BOARD_HEIGHT - 1 - y
            elif p == BLACK_KING:
                if CENTER_MAP[y][x] == 1:
                    black_center += 10
            # for dx, dy in defensive_directions:
            #     def_x, def_y = x + dx, y + dy
            #     if not Board.is_valid(def_x, def_y):
            #         continue
            #     def_p = self.board[def_x, def_y]
            #     if def_p == EMPTY:
            #         continue
            #     if def_p in OPPONENTS[p]:
            #         continue
            #     if p in RED_TEAM:
            #         red_defensive_count += 1
            #     else:
            #         black_defensive_count += 1

    ans = 10 * difference + \
          5 * red_center - black_center + \
          5 * red_to_king_distance - black_to_king_distance  # + \
    # 10 * (red_defensive_count - black_defensive_count)
    eval_cache[self] = ans
    return ans


evaluate = evaluate2 if USE_EVAL2 else evaluate1


states_expanded = 0

class State:
    board: Board
    current_player: int
    alpha: int
    beta: int
    depth: int

    def __init__(self, board: Board, player: int, alpha: int, beta: int, depth: int):
        self.board = board
        self.current_player = player
        self.alpha = alpha
        self.beta = beta
        self.depth = depth

    def get_next_move(self) -> Tuple[int, Optional[Board]]:
        """
        Performs an alpha-beta search to find the best next move.
        :return: the best next move based on this state, or None if no move is possible
        """
        if self in cache:
            if cache[self][0] <= self.depth and cache[self][1] == EXACT:
                return tuple(cache[self][2])
            t, c = cache[self][1], cache[self][2]
            if t == LOWER_BOUND:
                if c[0] > self.beta:
                    return c[0], c[1]
            else:
                if c[0] < self.alpha:
                    return c[0], c[1]
        winner = self.get_winner()
        if winner != 0:
            # +INF for RED, -INF for BLACK
            utility = winner * INFINITY // (self.depth + 1)
            self.save_to_cache(EXACT, utility, None)
            return utility, None
        if self.depth == DEPTH_LIMIT:
            return evaluate(self), None
        global states_expanded
        states_expanded += 1
        # -INF for RED, INF for BLACK
        best_utility = -self.current_player * INFINITY
        successors = self.get_successors()
        # no possible moves
        if not successors:
            # we set it to behave that both players don't want this situation
            self.save_to_cache(EXACT, best_utility, None)
            return best_utility, None
        best_move = successors[0].board
        if self.current_player == RED_PLAYER:
            # MAX node
            for successor in successors:
                successor.alpha = self.alpha
                successor.beta = self.beta
                u, _ = successor.get_next_move()
                if u > best_utility:
                    best_utility, best_move = u, successor.board
                if u > self.beta:
                    #                    None because we have pruned the current node
                    self.save_to_cache(LOWER_BOUND, best_utility, None)
                    return best_utility, None
                self.alpha = max(self.alpha, u)
        else:
            # MIN node
            for successor in successors:
                successor.alpha = self.alpha
                successor.beta = self.beta
                u, _ = successor.get_next_move()
                if u < best_utility:
                    best_utility, best_move = u, successor.board
                if u < self.alpha:
                    self.save_to_cache(UPPER_BOUND, best_utility, None)
                    return best_utility, None
                self.beta = min(self.beta, u)
        self.save_to_cache(EXACT, best_utility, best_move)
        return best_utility, best_move

    def get_successors(self) -> List[State]:
        ans = [State(a, -self.current_player, self.alpha, self.beta, self.depth + 1)
               for a in self.get_actions()]
        # increasing if MIN (black), decreasing if MAX (red)
        ans = sorted(ans, reverse=bool(self.current_player + 1))
        return ans

    def save_to_cache(self, type: int, utility: int, move: Optional[Board]):
        cache[self] = [self.depth, type, [utility, move]]

    def get_actions(self) -> List[Board]:
        return self.board.get_moves(self.current_player)

    def get_winner(self) -> int:
        return self.board.get_winner()

    def __lt__(self, other):
        # if not isinstance(other, self.__class__):
        #     return False
        if USE_EVAL_ORDERING:
            return evaluate(self) < evaluate(other)
        else:
            return self.board.__lt__(other.board)

    def __hash__(self):
        return self.board.__hash__()

    def __eq__(self, other):
        # if not isinstance(other, self.__class__):
        #     return False
        return self.board.__eq__(other.board)

    def __str__(self):
        return self.board.__str__()

    def __repr__(self):
        return self.__str__()


def main(input_file_path: str, output_file_path: str) -> Tuple[List, float]:
    start_time = time.time()
    player = RED_PLAYER
    initial_state = State(Board(input_file=input_file_path), player, -INFINITY, INFINITY, 0)
    ans = [initial_state]
    count = 1
    if LOGGING:
        print('Game started!')
        print('===================================================')
    move = initial_state.get_next_move()[1]
    if LOGGING:
        print('Move #{}:\n'.format(count))
        print(move)
        print('Time consumed so far: {} seconds'.format(time.time() - start_time))
        print('===================================================')
    while move:
        # if count > 100:
        #     print('Game aborted! Too many moves!')
        #     return
        count += 1
        ans.append(move)
        player = -player
        move = State(move, player, -INFINITY, INFINITY, 0).get_next_move()[1]
        if LOGGING:
            print('Move #{}:\n'.format(count))
            print(move)
            print('Time consumed so far: {} seconds'.format(time.time() - start_time))
            print('===================================================')
    if LOGGING:
        print('Game finished in {} moves!'.format(len(ans) - 1))
        print('Writing solutions...')
    with open(output_file_path, 'w') as file:
        for state in ans:
            file.write(state.__str__() + '\n')
    end_time = time.time()
    elapsed = end_time - start_time
    if LOGGING:
        print('Total time consumed: {} seconds'.format(elapsed))
        print('Total states expanded: {}'.format(states_expanded))
    if not RUN_ALL_TESTS:
        exit()
    return ans, elapsed


def test(*_, **__):
    initial_state = State(Board(input_file='checkers2.txt'), RED_PLAYER, -INFINITY, INFINITY, 0)
    print(initial_state.get_next_move())


def run_all_tests():
    print(f'Running tests on depth limit = {DEPTH_LIMIT}')
    tests = ['checkers1.txt', 'checkers2.txt', 'checkers3.txt', 'checkers4.txt', 'checkers7.txt']
    moves = [3, 7, 7, 9, 13, 5, 15]
    for t, m in zip(tests, moves):
        print(f'Running test {t}...')
        a, elapsed = main(t, 'o.txt')
        print(f'[{len(a) - 1 == m}] Test {t} finished in {elapsed} seconds '
              f'with {len(a) - 1}/{m} moves!')
        cache.clear()
        eval_cache.clear()


def init() -> Tuple[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()
    return args.inputfile, args.outputfile


if __name__ == '__main__':
    if RUN_ALL_TESTS:
        run_all_tests()
        exit()
    f = test if TESTING else main
    if not PROFILE:
        f(*init())
    else:
        import cProfile
        import os

        cProfile.run("f(*init())", 'time.prof')
        os.system('snakeviz time.prof')
