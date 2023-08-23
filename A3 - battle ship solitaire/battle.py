from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict
from typing import List, Set, Tuple, Optional, Dict

start_time = time.time()

# debugging constants
PROFILE = False
TESTING = False
DEBUG = False

EMPTY = '0'
WATER = '.'
# the 1 * 1 piece
SUBMARINE = 'S'
LEFT_PART = '<'
RIGHT_PART = '>'
TOP_PART = '^'
BOTTOM_PART = 'v'
MIDDLE_PART = 'M'

# variable constants
UNDETERMINED_LOCATION = (-1, -1)
UNDETERMINED_ORIENTATION = -1
SUBMARINE_ORIENTATION = 0
HORIZONTAL_ORIENTATION = 1
VERTICAL_ORIENTATION = 2

Location = Tuple[int, int]
#            coordinate             orientation
ShipValue = Tuple[Location, int]


class ShipVariable:
    # (-1, -1) for undetermined
    location: Location
    orientation: int
    length: int

    # undetermined and all possible locations and orientations of the top-left corner of the ship
    domain: Set[ShipValue]
    default_domain: Set[ShipValue]
    pruned_values: Dict[Tuple[ShipVariable, ShipValue], Set[ShipValue]]
    is_constant: bool

    def __init__(self, coordinate: Location, orientation: int, length: int, domain: Set[ShipValue]):
        self.location = coordinate
        self.orientation = orientation
        self.length = length
        self.domain = domain.copy()
        self.default_domain = domain.copy()
        self.pruned_values = defaultdict(set)
        self.is_constant = False

    def shrink_domain(self, shrink_by: Set[ShipValue],
                      cause_var: ShipVariable, cause_val: ShipValue) -> bool:
        if DEBUG:
            assert not self.is_constant
        shrink_by = shrink_by & self.domain
        if len(shrink_by) == 0:
            return False
        self.domain -= shrink_by
        self.pruned_values[(cause_var, cause_val)] |= shrink_by
        return True

    def restore_domain(self, cause_var: ShipVariable, cause_val: ShipValue):
        self.domain |= self.pruned_values[(key := (cause_var, cause_val))]
        self.pruned_values.pop(key)

    def shrink_default_domain(self, shrink_by: Set[ShipValue]):
        if DEBUG:
            assert not self.is_constant
        self.default_domain -= shrink_by
        self.restore_default_domain()

    def set_constant(self, coordinate: Location = None, orientation: int = None):
        if self.is_constant:
            return
        self.is_constant = True
        if coordinate is None:
            coordinate = self.location
        else:
            self.location = coordinate
        if orientation is not None:
            self.orientation = orientation
        self.set_default_domain({(coordinate, orientation)})

    def set_domain(self, new_domain: Set[ShipValue]):
        if DEBUG:
            assert not self.is_constant
        self.domain = new_domain.copy()

    def set_default_domain(self, new_domain: Set[ShipValue]):
        self.default_domain = new_domain.copy()
        self.restore_default_domain()

    def restore_default_domain(self):
        self.domain = self.default_domain.copy()

    def get_value(self) -> ShipValue:
        return self.location, self.orientation

    def assign(self, value: ShipValue):
        if DEBUG:
            assert not self.is_constant
        self.location, self.orientation = value

    def unassign(self):
        if DEBUG:
            assert not self.is_constant
        self.location = UNDETERMINED_LOCATION
        self.orientation = UNDETERMINED_ORIENTATION

    def is_assigned(self) -> bool:
        if DEBUG:
            if self.location != UNDETERMINED_LOCATION:
                assert self.orientation != UNDETERMINED_ORIENTATION
        return self.location != UNDETERMINED_LOCATION and \
            self.orientation != UNDETERMINED_ORIENTATION

    def __eq__(self, other):
        if isinstance(other, ShipVariable):
            return self.location == other.location and \
                self.orientation == other.orientation and self.length == other.length
        elif isinstance(other, str):
            if other == SUBMARINE:
                return self.orientation == SUBMARINE_ORIENTATION
            if other == TOP_PART:
                return self.orientation == VERTICAL_ORIENTATION
            if other == LEFT_PART:
                return self.orientation == HORIZONTAL_ORIENTATION
        return False

    def __hash__(self):
        return hash(id(self))

    def __str__(self):
        return f'location: {self.location} orientation: {self.orientation} len: {self.length} ' \
               f'in {self.domain}'

    def __repr__(self):
        return self.__str__()


class CSP:
    grid: List[List[str]]
    # the board after preprocessed
    original_grid: Tuple[Tuple[str]]
    # grid used for filling water and restore
    backup_grid: Optional[List[List[str]]] = None
    width: int
    height: int

    row_constraints: List[int]
    column_constraints: List[int]
    ship_constraints: List[int]
    # a list of the same length as ship_constraints (same structure)
    # we have ship_constraints[0] number of submarine
    variables: List[List[ShipVariable]]

    def __init__(self, filename: str):
        self.read_from_file(filename)
        self.preprocess()

    def is_solved(self):
        return self.is_row_column_valid() and self.is_ships_valid()

    def is_complete(self) -> bool:
        return all(all(v.is_assigned() for v in vs) for vs in self.variables)

    def assign_value(self, var: ShipVariable, value: ShipValue) -> bool:
        var.assign(value)
        return self.try_place_ship(var)

    def unassign(self, var: ShipVariable):
        self.remove_ship(var)
        var.unassign()

    def select_next_variable(self) -> ShipVariable:
        for vars in self.variables[::-1]:
            for var in vars:
                if not var.is_assigned():
                    return var

    def is_surrounded_by_water(self, x: int, y: int, orientation: Tuple[int, int] = (1, 0)) -> bool:
        """Precondition: self[x, y] not in [EMPTY, WATER] """
        p = self[x, y]
        directions = []
        if p == SUBMARINE:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif p == TOP_PART:
            directions = [(1, 0), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif p == BOTTOM_PART:
            directions = [(1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif p == LEFT_PART:
            directions = [(-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif p == RIGHT_PART:
            directions = [(1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif p == MIDDLE_PART:
            directions = [(1, 1), (-1, 1), (-1, -1), (1, -1),
                          (orientation[1], orientation[0]), (-orientation[1], orientation[0])]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not self.is_valid_index(nx, ny):
                continue
            if self[nx, ny] != WATER:
                return False
        return True

    def is_ships_valid(self) -> bool:
        visited_middles = set()
        ships = [0] * len(self.ship_constraints)
        for y in range(self.height):
            for x in range(self.width):
                if (p := self[x, y]) == EMPTY:
                    return False
                if p == WATER:
                    continue
                if p == MIDDLE_PART and (x, y) not in visited_middles:
                    # "floating" middles
                    return False
                if p == SUBMARINE:
                    if not self.is_surrounded_by_water(x, y):
                        return False
                    ships[0] += 1
                    if ships[0] > self.ship_constraints[0]:
                        return False
                elif p == LEFT_PART:
                    if not self.is_surrounded_by_water(x, y):
                        return False
                    i = x + 1
                    if i == self.width:
                        return False
                    exhaust = False
                    while self[(t := (i, y))] == MIDDLE_PART:
                        if not self.is_surrounded_by_water(i, y, (1, 0)):
                            return False
                        i += 1
                        visited_middles.add(t)
                        if i == self.width:
                            exhaust = True
                            break
                    if exhaust:
                        return False
                    if self[i, y] != RIGHT_PART:
                        return False
                    if (l := i - x) >= len(self.ship_constraints):
                        return False
                    ships[l] += 1
                    if ships[l] > self.ship_constraints[l]:
                        return False
                elif p == TOP_PART:
                    if not self.is_surrounded_by_water(x, y):
                        return False
                    i = y + 1
                    if i == self.height:
                        return False
                    exhaust = False
                    while self[(t := (x, i))] == MIDDLE_PART:
                        if not self.is_surrounded_by_water(x, i, (0, 1)):
                            return False
                        i += 1
                        visited_middles.add(t)
                        if i == self.height:
                            exhaust = True
                            break
                    if exhaust:
                        return False
                    if self[x, i] != BOTTOM_PART:
                        return False
                    if (l := i - y) >= len(self.ship_constraints):
                        return False
                    ships[l] += 1
                    if ships[l] > self.ship_constraints[l]:
                        return False
        return ships == self.ship_constraints

    def is_row_column_valid(self) -> bool:
        row_parts, column_parts = [0] * self.width, [0] * self.height
        for y in range(self.height):
            for x in range(self.width):
                if (p := self[x, y]) == EMPTY:
                    return False
                if p == WATER:
                    continue
                row_parts[y] += 1
                if row_parts[y] > self.row_constraints[y]:
                    return False
                column_parts[x] += 1
                if column_parts[x] > self.column_constraints[x]:
                    return False
        return row_parts == self.row_constraints and column_parts == self.column_constraints

    def get_row_column_parts(self) -> Tuple[List[int], List[int]]:
        # TODO: improve this implementation so that O(1)
        row_parts, column_parts = [0] * self.height, [0] * self.width
        for y in range(self.height):
            for x in range(self.width):
                if self[x, y] == EMPTY or self[x, y] == WATER:
                    continue
                row_parts[y] += 1
                column_parts[x] += 1
        return row_parts, column_parts

    def place_ship(self, var: ShipVariable):
        x, y = var.location
        l = var.length
        if var.orientation == HORIZONTAL_ORIENTATION:
            self[x, y] = LEFT_PART
            for i in range(x + 1, x + l - 1):
                self[i, y] = MIDDLE_PART
            self[x + l - 1, y] = RIGHT_PART
        elif var.orientation == VERTICAL_ORIENTATION:
            self[x, y] = TOP_PART
            for i in range(y + 1, y + l - 1):
                self[x, i] = MIDDLE_PART
            self[x, y + l - 1] = BOTTOM_PART
        else:
            self[x, y] = SUBMARINE

    def remove_ship(self, var: ShipVariable):
        x, y = var.location
        l = var.length
        if var.orientation == HORIZONTAL_ORIENTATION:
            for i in range(x, x + l):
                self[i, y] = self.original_grid[y][i]
        elif var.orientation == VERTICAL_ORIENTATION:
            for i in range(y, y + l):
                self[x, i] = self.original_grid[i][x]
        else:
            self[x, y] = self.original_grid[y][x]

    def try_place_ship(self, var: ShipVariable) -> bool:
        if DEBUG:
            assert var.location != UNDETERMINED_LOCATION and var.orientation != UNDETERMINED_LOCATION
        row_parts, column_parts = self.get_row_column_parts()
        x, y = var.location
        l = var.length
        c = self[x, y]
        if var.orientation == HORIZONTAL_ORIENTATION:
            if x + l > self.width:
                return False
            # if there is an empty in the potential space
            has_empty = False
            delta_row_parts = 0
            delta_column_parts = [0] * self.width
            # the first cell
            if c == EMPTY:
                delta_row_parts += 1
                delta_column_parts[x] += 1
                has_empty = True
            elif c != LEFT_PART:
                return False
            # middle cells
            for i in range(x + 1, x + l - 1):
                if (cc := self[i, y]) == EMPTY:
                    delta_row_parts += 1
                    delta_column_parts[i] += 1
                    has_empty = True
                elif cc != MIDDLE_PART:
                    return False
            # the last cell
            if (cc := self[x + l - 1, y]) == RIGHT_PART:
                # we have found a whole ship!
                if not has_empty:
                    return False
            elif cc != EMPTY:
                return False
            else:
                delta_row_parts += 1
                delta_column_parts[x + l - 1] += 1
            # now we can check the row and column constraint
            if row_parts[y] + delta_row_parts > self.row_constraints[y]:
                return False
            if any(column_parts[i] + delta_column_parts[i] > self.column_constraints[i]
                   for i in range(self.width)):
                return False
            # GOOD
            self.place_ship(var)
            return True
        elif var.orientation == VERTICAL_ORIENTATION:
            if y + l > self.height:
                return False
            has_empty = False
            delta_row_parts = [0] * self.height
            delta_column_parts = 0
            if c == EMPTY:
                delta_row_parts[y] += 1
                delta_column_parts += 1
                has_empty = True
            elif c != TOP_PART:
                return False
            for i in range(y + 1, y + l - 1):
                if (cc := self[x, i]) == EMPTY:
                    delta_row_parts[i] += 1
                    delta_column_parts += 1
                    has_empty = True
                elif cc != MIDDLE_PART:
                    return False
            if (cc := self[x, y + l - 1]) == BOTTOM_PART:
                if not has_empty:
                    return False
            elif cc != EMPTY:
                return False
            else:
                delta_row_parts[y + l - 1] += 1
                delta_column_parts += 1
            if column_parts[x] + delta_column_parts > self.column_constraints[x]:
                return False
            if any(row_parts[i] + delta_row_parts[i] > self.row_constraints[i]
                   for i in range(self.height)):
                return False
            self.place_ship(var)
            return True
        else:
            if row_parts[y] + 1 > self.row_constraints[y]:
                return False
            if column_parts[x] + 1 > self.column_constraints[x]:
                return False
            if self[x, y] == EMPTY:
                self.place_ship(var)
                return True
            return False

    def get_directions_to_fill_water(self, c: str):
        directions_to_fill = []
        if c == SUBMARINE:
            directions_to_fill = \
                [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif c == LEFT_PART:
            directions_to_fill = \
                [(-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif c == RIGHT_PART:
            directions_to_fill = \
                [(1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif c == TOP_PART:
            directions_to_fill = \
                [(1, 0), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif c == BOTTOM_PART:
            directions_to_fill = \
                [(1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        elif c == MIDDLE_PART:
            directions_to_fill = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        return directions_to_fill

    def forward_check(self, var: ShipVariable):
        x, y = var.location
        l = var.length
        orientation = var.orientation
        restores = set()
        # ships are surrounded by water
        if orientation == HORIZONTAL_ORIENTATION:
            locations = [(i, y) for i in range(x, x + l)]
        elif orientation == VERTICAL_ORIENTATION:
            locations = [(x, i) for i in range(y, y + l)]
        else:
            locations = [(x, y)]
        for i, j in locations:
            c = self[i, j]
            directions_fill = self.get_directions_to_fill_water(c)
            shrink_locations = [(i + dx, j + dy) for dx, dy in directions_fill] + locations
            shrink_domain = set(itertools.product(
                shrink_locations,
                (SUBMARINE_ORIENTATION, HORIZONTAL_ORIENTATION, VERTICAL_ORIENTATION)))
            for vars in self.variables:
                for v in vars:
                    if not v.is_assigned():
                        if v.shrink_domain(shrink_domain, var, var.get_value()):
                            restores.add(v)

        # note that row and column FC are so buggy and complex, we are good enough
        # the bug is that if we simply remove the whole row in the domain, this might
        # include incomplete part like a left part
        # row and column
        # row_parts, column_parts = self.get_row_column_parts()
        # for h in range(self.height):
        #     if row_parts[h] == self.row_constraints[h]:
        #         # the whole row cannot place ships
        #         shrink_locations = [(i, h) for i in range(self.width)]
        #         shrink_domain = set(itertools.product(
        #             shrink_locations,
        #             (SUBMARINE_ORIENTATION, HORIZONTAL_ORIENTATION, VERTICAL_ORIENTATION)))
        #         for vars in self.variables:
        #             for v in vars:
        #                 if not v.is_assigned():
        #                     if v.shrink_domain(shrink_domain, var, var.get_value()):
        #                         restores.add(v)
        # for w in range(self.width):
        #     if column_parts[w] == self.column_constraints[w]:
        #         shrink_locations = [(w, i) for i in range(self.height)]
        #         shrink_domain = set(itertools.product(
        #             shrink_locations,
        #             (SUBMARINE_ORIENTATION, HORIZONTAL_ORIENTATION, VERTICAL_ORIENTATION)))
        #         for vars in self.variables:
        #             for v in vars:
        #                 if not v.is_assigned():
        #                     if v.shrink_domain(shrink_domain, var, var.get_value()):
        #                         restores.add(v)
        return restores

    def fill_water(self):
        if DEBUG:
            assert self.backup_grid is None
        self.backup_grid = [[c for c in row] for row in self.grid]
        for y in range(self.height):
            for x in range(self.width):
                if self[x, y] == EMPTY:
                    self[x, y] = WATER

    def clear_water(self):
        if DEBUG:
            assert self.backup_grid is not None
        self.grid = self.backup_grid
        self.backup_grid = None

    def preprocess(self):
        # a level-1 preprocess, if we go deeper, then it becomes a GAC on cell variables
        # ships are surrounded by water
        for y in range(self.height):
            for x in range(self.width):
                if (c := self[x, y]) == EMPTY or c == WATER:
                    continue
                directions_to_fill = []
                if c == SUBMARINE:
                    directions_to_fill = \
                        [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
                elif c == LEFT_PART:
                    directions_to_fill = \
                        [(-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
                    # we don't set it to the MIDDLE_PART because we are unsure of if it is a RIGHT
                    # but we're sure that there are certain parts here, so we still fill water
                    # self[x + 1, y] = MIDDLE_PART
                    directions_to_fill.extend([(2, -1), (2, 1)])
                elif c == RIGHT_PART:
                    directions_to_fill = \
                        [(1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
                    # self[x - 1, y] = MIDDLE_PART
                    directions_to_fill.extend([(-2, -1), (-2, 1)])
                elif c == TOP_PART:
                    directions_to_fill = \
                        [(1, 0), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
                    # self[x, y + 1] = MIDDLE_PART
                    directions_to_fill.extend([(1, 2), (-1, 2)])
                elif c == BOTTOM_PART:
                    directions_to_fill = \
                        [(1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
                    # self[x, y - 1] = MIDDLE_PART
                    directions_to_fill.extend([(1, -2), (-1, -2)])
                elif c == MIDDLE_PART:
                    directions_to_fill = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for dx, dy in directions_to_fill:
                    nx, ny = x + dx, y + dy
                    if self.is_valid_index(nx, ny) and self[nx, ny] == EMPTY:
                        self[nx, ny] = WATER
        # fill rows where the # parts == row constraint
        row_parts, column_parts = self.get_row_column_parts()
        for y in range(self.height):
            if row_parts[y] == self.row_constraints[y]:
                for x in range(self.width):
                    if self[x, y] == EMPTY:
                        self[x, y] = WATER
        for x in range(self.width):
            if column_parts[x] == self.column_constraints[x]:
                for y in range(self.height):
                    if self[x, y] == EMPTY:
                        self[x, y] = WATER

        # set all submarine variables and prune some of other's domain
        for y in range(self.height):
            for x in range(self.width):
                c = self[x, y]
                if c in [EMPTY, LEFT_PART, TOP_PART]:
                    continue
                for vars in self.variables:
                    for var in vars:
                        if not var.is_assigned():
                            var.shrink_default_domain(
                                {((x, y), SUBMARINE_ORIENTATION), ((x, y), HORIZONTAL_ORIENTATION),
                                 ((x, y), VERTICAL_ORIENTATION)})
                if c == SUBMARINE:
                    for s in self.variables[0]:
                        if s.is_assigned():
                            continue
                        s.set_constant((x, y), SUBMARINE_ORIENTATION)
                        break

    def is_valid_index(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def read_from_file(self, filename: str):
        with open(filename) as file:
            raw = file.readlines()
            self.row_constraints = [int(c) for c in raw[0].strip()]
            self.column_constraints = [int(c) for c in raw[1].strip()]
            self.ship_constraints = [int(c) for c in raw[2].strip()]

            self.grid = [[c for c in raw[i].strip()] for i in range(3, len(raw))]
            self.original_grid = tuple(tuple(c for c in raw[i].strip()) for i in range(3, len(raw)))
            self.height = len(self.grid)
            self.width = len(self.grid[0])
            self.column_constraints = self.column_constraints
            self.row_constraints = self.row_constraints
            self.ship_constraints = self.ship_constraints

            self.variables = []
            all_locations = list(itertools.product(range(self.height), range(self.width)))
            # init submarine variables
            submarine_domain = {(location, SUBMARINE_ORIENTATION) for location in all_locations}
            self.variables.append([])
            for j in range(self.ship_constraints[0]):
                self.variables[0].append(ShipVariable(
                    UNDETERMINED_LOCATION, UNDETERMINED_ORIENTATION, 1, submarine_domain.copy()))
            # other variables
            domain = set(itertools.product(all_locations,
                                           (HORIZONTAL_ORIENTATION, VERTICAL_ORIENTATION)))
            for i in range(1, len(self.ship_constraints)):
                self.variables.append([])
                for j in range(self.ship_constraints[i]):
                    self.variables[i].append(ShipVariable(
                        UNDETERMINED_LOCATION, UNDETERMINED_ORIENTATION, i + 1, domain.copy()))

    def write_solution(self, filename: str):
        if DEBUG:
            assert self.is_solved()
        with open(filename, 'w') as file:
            s = ''.join(''.join(line) + '\n' for line in self.grid)
            file.write(s)

    def __setitem__(self, key: Tuple[int, int], value: str):
        x, y = key
        self.grid[y][x] = value

    def __getitem__(self, item: Tuple[int, int]) -> str:
        x, y = item
        return self.grid[y][x]

    def __str__(self):
        s = '\n ' + ''.join([str(c) for c in self.column_constraints]) + '\n'
        for y in range(self.height):
            s += str(self.row_constraints[y]) + ''.join(self.grid[y]) + '\n'
        return s

    def __repr__(self):
        return self.__str__()


def backtrack(csp: CSP) -> Optional[CSP]:
    if csp.is_complete():
        csp.fill_water()
        # print(csp)
        if csp.is_solved():
            return csp
        csp.clear_water()
        return None
    var = csp.select_next_variable()
    for value in var.domain:
        if not csp.assign_value(var, value):
            var.unassign()
            continue
        # do FC
        restores = csp.forward_check(var)
        if DEBUG:
            assert all((var, value) in v.pruned_values for v in restores)
        # print(csp)
        result = backtrack(csp)
        if result:
            return result
        csp.unassign(var)
        for restore_var in restores:
            restore_var.restore_domain(var, value)
    return None


def main(input_filename: str, output_filename: str):
    print(f'Start solving {input_filename}!')
    csp = CSP(input_filename)
    print('Preprocessed board:')
    print(csp)
    print('===================================================')
    csp = backtrack(csp)
    if csp is None:
        print('Failed to solve the puzzle!')
        exit(1)
    csp.write_solution(output_filename)
    print('===================================================')
    print('Solved! Result:')
    print(csp)
    end_time = time.time()
    print(f'Consumed {end_time - start_time} seconds!')
    exit(0)


def test(*_, **__):
    pass


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
    f = main if not TESTING else test
    if PROFILE:
        import cProfile
        import os

        cProfile.run("f(*init())", 'time.prof')
        os.system('snakeviz time.prof')
    else:
        f(*init())
