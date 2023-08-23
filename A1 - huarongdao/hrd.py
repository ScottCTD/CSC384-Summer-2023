# Author: Scott Cui
# Things to improve are marked with TODO
from __future__ import annotations

import copy
import heapq
import time
from argparse import ArgumentParser
from typing import Tuple, List, Optional, Callable, Dict, Set

# some constants
BOARD_WIDTH, BOARD_HEIGHT = 4, 5
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
SYMBOL_TO_SIZE = {
    '.': (1, 1),
    '2': (1, 1),
    '1': (2, 2),
    '^': (1, 2),
    '<': (2, 1)
}


class Piece:
    size: Tuple[int, int]
    coordinate: Tuple[int, int]
    is_empty: bool

    def __init__(self, size: Tuple[int, int], coordinate: Tuple[int, int], is_empty: bool) -> None:
        self.size = size
        self.coordinate = coordinate
        self.is_empty = is_empty

    def __repr__(self):
        s = ''
        if self.size == (1, 1):
            if self.is_empty:
                s = '.'
            else:
                s = '2'
        elif self.size == (1, 2):
            s = '<>'
        elif self.size == (2, 1):
            s = '^v'
        elif self.size == (2, 2):
            s = '1111'
        return s + f'\t{self.size} {self.coordinate} {self.is_empty}'


class Board:
    # TODO: clear up move logic

    grid: List[List[str]]

    def __init__(self, pieces: List[Piece] = None) -> None:
        if pieces is not None:
            self.grid = [[''] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
            self._construct_grid(pieces)

    def move_2x2(self, x: int, y: int, direction: Tuple[int, int]) -> Optional[Board]:
        # precondition: x and y represent the top-left
        dx, dy = direction
        new_grid = None
        if dx == -1:
            if x - 1 < 0:
                return None
            if self.grid[y][x - 1] != '.' or self.grid[y + 1][x - 1] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y][x + 1], new_grid[y][x - 1] = new_grid[y][x - 1], new_grid[y][x + 1]
            new_grid[y + 1][x + 1], new_grid[y + 1][x - 1] = \
                new_grid[y + 1][x - 1], new_grid[y + 1][x + 1]
        elif dy == -1:
            if y - 1 < 0:
                return None
            if self.grid[y - 1][x] != '.' or self.grid[y - 1][x + 1] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y + 1][x], new_grid[y - 1][x] = new_grid[y - 1][x], new_grid[y + 1][x]
            new_grid[y + 1][x + 1], new_grid[y - 1][x + 1] = \
                new_grid[y - 1][x + 1], new_grid[y + 1][x + 1]
        elif dx == 1:
            if x + 2 >= BOARD_WIDTH:
                return None
            if self.grid[y][x + 2] != '.' or self.grid[y + 1][x + 2] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y][x], new_grid[y][x + 2] = new_grid[y][x + 2], new_grid[y][x]
            new_grid[y + 1][x], new_grid[y + 1][x + 2] = \
                new_grid[y + 1][x + 2], new_grid[y + 1][x]
        elif dy == 1:
            if y + 2 >= BOARD_HEIGHT:
                return None
            if self.grid[y + 2][x] != '.' or self.grid[y + 2][x + 1] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y][x], new_grid[y + 2][x] = new_grid[y + 2][x], new_grid[y][x]
            new_grid[y][x + 1], new_grid[y + 2][x + 1] = \
                new_grid[y + 2][x + 1], new_grid[y][x + 1]
        else:
            raise ValueError('Incorrect direction!')
        new_board = Board()
        new_board.grid = new_grid
        return new_board

    def move_1x1(self, x: int, y: int, direction: Tuple[int, int]) -> Optional[Board]:
        dx, dy = direction
        new_x, new_y = x + dx, y + dy
        if not (0 <= new_x < BOARD_WIDTH and 0 <= new_y < BOARD_HEIGHT):
            return None
        if self.grid[new_y][new_x] != '.':
            return None
        new_grid = copy.deepcopy(self.grid)
        new_grid[y][x], new_grid[new_y][new_x] = new_grid[new_y][new_x], new_grid[y][x]
        new_board = Board()
        new_board.grid = new_grid
        return new_board

    def move_1x2(self, x: int, y: int, direction: Tuple[int, int]) -> Optional[Board]:
        dx, dy = direction
        new_grid = None
        if dy == 0:
            new_x = x + dx
            if not (0 <= new_x < BOARD_WIDTH):
                return None
            if self.grid[y][new_x] != '.' or self.grid[y + 1][new_x] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y][x], new_grid[y][new_x] = new_grid[y][new_x], new_grid[y][x]
            new_grid[y + 1][x], new_grid[y + 1][new_x] = new_grid[y + 1][new_x], new_grid[y + 1][x]
        elif dy == -1:
            if y - 1 < 0:
                return None
            if self.grid[y - 1][x] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y][x], new_grid[y - 1][x] = new_grid[y - 1][x], new_grid[y][x]
            new_grid[y + 1][x], new_grid[y][x] = new_grid[y][x], new_grid[y + 1][x]
        elif dy == 1:
            if y + 2 >= BOARD_HEIGHT:
                return None
            if self.grid[y + 2][x] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y + 1][x], new_grid[y + 2][x] = new_grid[y + 2][x], new_grid[y + 1][x]
            new_grid[y][x], new_grid[y + 1][x] = new_grid[y + 1][x], new_grid[y][x]
        else:
            raise ValueError('Incorrect direction!')
        new_board = Board()
        new_board.grid = new_grid
        return new_board

    def move_1x2_offset(self, x: int, y: int, direction: Tuple[int, int]) -> Optional[Board]:
        return self.move_1x2(x, y - 1, direction)

    def move_2x1(self, x: int, y: int, direction: Tuple[int, int]) -> Optional[Board]:
        dx, dy = direction
        new_grid = None
        if dx == 0:
            new_y = y + dy
            if not (0 <= new_y < BOARD_HEIGHT):
                return None
            if self.grid[new_y][x] != '.' or self.grid[new_y][x + 1] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y][x], new_grid[new_y][x] = new_grid[new_y][x], new_grid[y][x]
            new_grid[y][x + 1], new_grid[new_y][x + 1] = new_grid[new_y][x + 1], new_grid[y][x + 1]
        elif dx == -1:
            if x - 1 < 0:
                return None
            if self.grid[y][x - 1] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y][x], new_grid[y][x - 1] = new_grid[y][x - 1], new_grid[y][x]
            new_grid[y][x + 1], new_grid[y][x] = new_grid[y][x], new_grid[y][x + 1]
        elif dx == 1:
            if x + 2 >= BOARD_WIDTH:
                return None
            if self.grid[y][x + 2] != '.':
                return None
            new_grid = copy.deepcopy(self.grid)
            new_grid[y][x + 1], new_grid[y][x + 2] = new_grid[y][x + 2], new_grid[y][x + 1]
            new_grid[y][x], new_grid[y][x + 1] = new_grid[y][x + 1], new_grid[y][x]
        else:
            raise ValueError('Incorrect direction!')
        new_board = Board()
        new_board.grid = new_grid
        return new_board

    def move_2x1_offset(self, x: int, y: int, direction: Tuple[int, int]) -> Optional[Board]:
        return self.move_2x1(x - 1, y, direction)

    def find_empty_pieces(self) -> List[Tuple[int, int]]:
        # TODO: use a variable keeping track of empty pieces, instead of iterate through the 
        #  whole board all the times
        r = []
        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH):
                if self.grid[i][j] == '.':
                    r.append((i, j))
        return r

    def _construct_grid(self, pieces: List[Piece]) -> None:
        for piece in pieces:
            i, j = piece.coordinate
            if piece.size == (1, 1):
                if piece.is_empty:
                    self.grid[i][j] = '.'
                else:
                    self.grid[i][j] = '2'
            elif piece.size == (2, 1):
                self.grid[i][j] = '<'
                self.grid[i][j + 1] = '>'
            elif piece.size == (1, 2):
                self.grid[i][j] = '^'
                self.grid[i + 1][j] = 'v'
            elif piece.size == (2, 2):
                self.grid[i][j] = '1'
                self.grid[i][j + 1] = '1'
                self.grid[i + 1][j] = '1'
                self.grid[i + 1][j + 1] = '1'
            else:
                raise ValueError('Invalid piece size')

    def __repr__(self) -> str:
        a = ''
        for line in self.grid:
            a += ''.join(line) + '\n'
        return a

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.grid == other.grid

    def __hash__(self) -> int:
        return hash(str(self.grid))


PIECE_TO_MOVE: Dict[str, Callable] = {
    '2': Board.move_1x1,
    '1': Board.move_2x2,
    '^': Board.move_1x2,
    'v': Board.move_1x2_offset,
    '<': Board.move_2x1,
    '>': Board.move_2x1_offset,
}

# used for the advanced heuristic function
PIECE_TO_WEIGHT: Dict[str, int] = {
    '1': 1,
    '^': 1,
    '<': 2,
    '>': 2
}


class State:
    parent: State
    board: Board
    g: int

    def __init__(self, board: Board, parent: State = None, g: int = 0) -> None:
        self.board = board
        self.parent = parent
        self.g = g

    def get_successors(self) -> Set[State]:
        # TODO: use some tricks to not add repetitive states, instead of using set
        empty_pieces = self.board.find_empty_pieces()
        r = set()
        done_2x2 = False
        for i, j in empty_pieces:
            for dx, dy in DIRECTIONS:
                y, x = i + dy, j + dx
                if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
                    continue
                p = self.board.grid[y][x]
                if p not in PIECE_TO_MOVE:
                    continue
                # special case for 2x2 block
                if p == '1':
                    if not done_2x2:
                        if x - 1 >= 0 and self.board.grid[y][x - 1] == '1':
                            x -= 1
                        if y - 1 >= 0 and self.board.grid[y - 1][x] == '1':
                            y -= 1
                        done_2x2 = True
                    else:
                        continue
                new_board = PIECE_TO_MOVE[p](self.board, x, y, (-dx, -dy))
                if new_board is not None:
                    r.add(State(new_board, self, self.g + 1))
        return r

    def f(self):
        return self.g + self.h()

    def h(self):
        return self.h_manhattan()

    def h_manhattan(self):
        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH):
                # must be the top left corner of the 2x2 block
                if self.board.grid[i][j] == '1':
                    # get the bottom left corner
                    i += 1
                    return BOARD_HEIGHT - i - 1 + abs(1 - j)

    def h_advanced(self):
        h_value = 0
        y, x = 0, 0
        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH):
                p = self.board.grid[i][j]
                # must be the top left corner of the 2x2 block
                if p == '1':
                    # get the bottom left corner
                    i += 1
                    h_value += BOARD_HEIGHT - i - 1 + abs(1 - j)
                    y, x = i, j
                    break
        for i in range(y + 1, BOARD_HEIGHT):
            for j in range(x, x + 2):
                p = self.board.grid[i][j]
                if p in PIECE_TO_WEIGHT:
                    h_value += PIECE_TO_WEIGHT[p]

        return h_value

    def is_goal(self):
        return self.board.grid[4][1] == '1' and self.board.grid[4][2] == '1'

    def get_solution(self) -> List[State]:
        solution = []
        current = self
        while current is not None:
            solution.append(current)
            current = current.parent
        return solution[::-1]

    def __repr__(self):
        return f'\n{self.board.__repr__()}g = {self.g} h = {self.h()} f = {self.f()}'

    def __hash__(self):
        return self.board.__hash__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.board == other.board

    def __lt__(self, other):
        return self.f() < other.f()


def dfs(initial_state: State) -> List[State]:
    stack = [initial_state]
    visited = set()
    count = 0
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        # print(s)
        count += 1
        visited.add(s)
        if s.is_goal():
            print(f'Expanded {count} states.')
            return s.get_solution()
        for child in s.get_successors():
            stack.append(child)
    return []


def a_star(initial_state: State) -> List[State]:
    queue = [initial_state]
    visited = set()
    count = 0
    while queue:
        s = heapq.heappop(queue)
        if s in visited:
            # do not do pruning when adding children, because that would not progress the
            # algorithm well
            continue
        # print(s)
        # print(f'{len(queue)} in queue')
        count += 1
        visited.add(s)
        if s.is_goal():
            print(f'Expanded {count} states.')
            return s.get_solution()
        successors = s.get_successors()
        # print(f'Generated states {len(a)}')
        for child in successors:
            heapq.heappush(queue, child)
    return []


def read_from_file(filename: str) -> Board:
    pieces = []
    with open(filename) as file:
        b = False
        for i, line in enumerate(file.readlines()):
            for j, c in enumerate(line):
                coord = (i, j)
                p = None
                if c == '.':
                    p = Piece((1, 1), coord, True)
                elif c == '2':
                    p = Piece((1, 1), coord, False)
                elif c == '1' and not b:
                    p = Piece((2, 2), coord, False)
                    b = True
                elif c == '<':
                    p = Piece((2, 1), coord, False)
                elif c == '^':
                    p = Piece((1, 2), coord, False)
                if p is not None:
                    pieces.append(p)
    return Board(pieces)


def write_to_file(filename: str, solution: List[State]):
    with open(filename, 'w') as file:
        for state in solution:
            file.write(state.board.__repr__() + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()
    input_file = args.inputfile
    output_file = args.outputfile
    algo = dfs if args.algo == 'dfs' else a_star

    board = read_from_file(input_file)
    initial_state = State(board)
    print(f'Starting solving the following initial configuration...')
    print(board)

    start_time = time.time()
    solution = algo(initial_state)
    end_time = time.time()
    print(f'Successfully found a solution with cost = {solution[-1].g} '
          f'in {end_time - start_time} seconds!')

    print('Saving the solution...')
    write_to_file(output_file, solution)
    print(f'The solution is saved in {output_file} with the specified format.')
