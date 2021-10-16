import os
import datetime, time
import argparse
import search as searchAlgo
import psutil

class SokobanState:
    # player: 2-tuple representing player location (coordinates)
    # boxes: list of 2-tuples indicating box locations
    def __init__(self, player, boxes):
        # self.data stores the state
        self.data = tuple([player] + sorted(boxes))
        # below are cache variables to avoid duplicated computation
        self.all_adj_cache = None
        self.adj = {}
        self.dead = None
        self.solved = None

    def __str__(self):
        return 'player: ' + str(self.player()) + ' boxes: ' + str(self.boxes())

    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data

    def __lt__(self, other):
        return self.data < other.data

    def __hash__(self):
        return hash(self.data)

    # return player location
    def player(self):
        return self.data[0]

    # return boxes locations
    def boxes(self):
        return self.data[1:]

    # check goal state
    def is_goal(self, problem):
        if self.solved is None:
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved

    #return: (whether a move is valid, whether a box is moved, the next state)
    def act(self, problem, act):
        if act in self.adj:
            return self.adj[act]
        else:
            val = problem.valid_move(self, act)
            self.adj[act] = val
            return val

    #Check dead state
    def box_is_cornered(self, map, box, targets, all_boxes):

        def row_is_trap(offset):
            target_count = 0
            box_count = 1
            for direction in [-1, 1]:
                index = box[1] + direction
                while not map[box[0]][index].wall:
                    if map[box[0] + offset][index].floor:
                        return None
                    elif map[box[0]][index].target:
                        target_count += 1
                    elif (box[0], index) in all_boxes:
                        box_count += 1
                    index += direction

            if box_count > target_count:
                return True
            return None

        def column_is_trap(offset):
            target_count = 0
            box_count = 1
            for direction in [-1, 1]:
                index = box[0] + direction
                while not map[index][box[1]].wall:
                    if map[index][box[1] + offset].floor:
                        return None
                    elif map[index][box[1]].target:
                        target_count += 1
                    elif (index, box[1]) in all_boxes:
                        box_count += 1
                    index += direction

            if box_count > target_count:
                return True
            return None

        # Literal corners
        if box not in targets:
            if map[box[0] - 1][box[1]].wall and map[box[0]][box[1] - 1].wall:
                return True
            elif map[box[0] - 1][box[1]].wall and map[box[0]][box[1] + 1].wall:
                return True
            elif map[box[0] + 1][box[1]].wall and map[box[0]][box[1] - 1].wall:
                return True
            elif map[box[0] + 1][box[1]].wall and map[box[0]][box[1] + 1].wall:
                return True

        # Expanded corners
            if map[box[0] - 1][box[1]].wall:
                if row_is_trap(offset=-1):
                    return True
            elif map[box[0] + 1][box[1]].wall:
                if row_is_trap(offset=1):
                    return True
            elif map[box[0]][box[1] - 1].wall:
                if column_is_trap(offset=-1):
                    return True
            elif map[box[0]][box[1] + 1].wall:
                if column_is_trap(offset=1):
                    return True

        return None

    def adj_box(self, box, all_boxes):
        adj = []
        for i in all_boxes:
            if box[0] - 1 == i[0] and box[1] == i[1]:
                adj.append({'box': i, 'direction': 'vertical'})
            elif box[0] + 1 == i[0] and box[1] == i[1]:
                adj.append({'box': i, 'direction': 'vertical'})
            elif box[1] - 1 == i[1] and box[0] == i[0]:
                adj.append({'box': i, 'direction': 'horizontal'})
            elif box[1] + 1 == i[1] and box[0] == i[0]:
                adj.append({'box': i, 'direction': 'horizontal'})
        return adj

    def box_is_trapped(self, map, box, targets, all_boxes):
        if self.box_is_cornered(map, box, targets, all_boxes):
            return True

        adj_boxes = self.adj_box(box, all_boxes)
        for i in adj_boxes:
            if box not in targets and i not in targets:
                if i['direction'] == 'vertical':
                    if map[box[0]][box[1] - 1].wall and map[i['box'][0]][i['box'][1] - 1].wall:
                        return True
                    elif map[box[0]][box[1] + 1].wall and map[i['box'][0]][i['box'][1] + 1].wall:
                        return True
                if i['direction'] == 'horizontal':
                    if map[box[0] - 1][box[1]].wall and map[i['box'][0] - 1][i['box'][1]].wall:
                        return True
                    elif map[box[0] + 1][box[1]].wall and map[i['box'][0] + 1][i['box'][1]].wall:
                        return True

        return None

    def deadp(self, problem):
        temp_boxes = self.data[1:]
        for box in list(temp_boxes):
            if self.box_is_trapped(problem.map, box, problem.targets, temp_boxes):
                self.dead = True
        return self.dead

    # return all adjacent states of one state
    def all_adj(self, problem):
        if self.all_adj_cache is None:
            succ = []
            for move in 'udlr':
                valid, box_moved, nextS = self.act(problem, move)
                if valid:
                    succ.append((move, nextS, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache


class MapTile:
    def __init__(self, wall=False, floor=False, target=False):
        self.wall = wall
        self.floor = floor
        self.target = target


def parse_move(move):
    if move == 'u':
        return (-1, 0)
    elif move == 'd':
        return (1, 0)
    elif move == 'l':
        return (0, -1)
    elif move == 'r':
        return (0, 1)
    raise Exception('Invalid move character.')


class DrawObj:
    WALL = '\033[37;47m \033[0m'
    PLAYER = '\033[97;40m@\033[0m'
    BOX_OFF = '\033[30;101mX\033[0m'
    BOX_ON = '\033[30;102mX\033[0m'
    TARGET = '\033[97;40m*\033[0m'
    FLOOR = '\033[30;40m \033[0m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class SokobanProblem(searchAlgo.SearchProblem):
    # valid sokoban characters
    valid_chars = '#@+$*. '

    def __init__(self, map, dead_detection=False):
        self.map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0, 0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.parse_map(map)

    # parse the input string into game map
    # Wall              #
    # Player            @
    # Player on target  +
    # Box               $
    # Box on target     *
    # Target            .
    # Floor             (space)
    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map) - 1, len(self.map[-1]) - 1)
        for c in input_str:
            if c == '#':
                self.map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.map[-1].append(MapTile(floor=True))
                self.init_player = coordinates()
            elif c == '+':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_player = coordinates()
                self.targets.append(coordinates())
            elif c == '$':
                self.map[-1].append(MapTile(floor=True))
                self.init_boxes.append(coordinates())
            elif c == '*':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_boxes.append(coordinates())
                self.targets.append(coordinates())
            elif c == '.':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.targets.append(coordinates())
            elif c == '\n':
                self.map.append([])
        assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        self.numboxes = len(self.init_boxes)

    def print_state(self, s):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                target = self.map[row][col].target
                box = (row, col) in s.boxes()
                player = (row, col) == s.player()
                if box and target:
                    print(DrawObj.BOX_ON, end='')
                elif player and target:
                    print(DrawObj.PLAYER, end='')
                elif target:
                    print(DrawObj.TARGET, end='')
                elif box:
                    print(DrawObj.BOX_OFF, end='')
                elif player:
                    print(DrawObj.PLAYER, end='')
                elif self.map[row][col].wall:
                    print(DrawObj.WALL, end='')
                else:
                    print(DrawObj.FLOOR, end='')
            print()

    # decide if a move is valid
    # return: (whether a move is valid, whether a box is moved, the next state)
    def valid_move(self, s, move, p=None):
        if p is None:
            p = s.player()
        dx, dy = parse_move(move)
        x1 = p[0] + dx
        y1 = p[1] + dy
        x2 = x1 + dx
        y2 = y1 + dy
        if self.map[x1][y1].wall:
            return False, False, None
        elif (x1, y1) in s.boxes():
            if self.map[x2][y2].floor and (x2, y2) not in s.boxes():
                return True, True, SokobanState((x1, y1),
                                                [b if b != (x1, y1) else (x2, y2) for b in s.boxes()])
            else:
                return False, False, None
        else:
            return True, False, SokobanState((x1, y1), s.boxes())

    # detect dead end
    def dead_end(self, s):
        if not self.dead_detection:
            return False

        return s.deadp(self)

    def start(self):
        return SokobanState(self.init_player, self.init_boxes)

    def goalp(self, s):
        return s.is_goal(self)

    def expand(self, s):
        if self.dead_end(s):
            return []
        return s.all_adj(self)

    def display_state(self, s):
        self.print_state(s)

class Heuristic:
    def __init__(self, problem):
        self.problem = problem
        self.buff = self.calc_cost()
        self.box_state = self.problem.init_boxes
        self.memo = dict()

    # Problem 3: Simple admissible heuristic
    def calc_manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    #simple heuristic function using manhattan distance
    def heuristic(self, s):
        box_pos = s.data[1:]
        targets = self.problem.targets
        targets_left = len(targets)
        total = 0
        for ind, box in enumerate(box_pos):
            total += self.calc_manhattan(box, targets[ind])
            if box in targets:
                targets_left -= 1
        return total * targets_left

    # find the minimum number of steps from one tile to its nearest target tile
    def calc_cost(self):

        def flood(x, y, cost):
            if not visited[x][y]:

                # Update cost if less than previous target
                if buff[x][y] > cost:
                    buff[x][y] = cost
                visited[x][y] = True

                # Check adjacent floors
                if self.problem.map[x - 1][y].floor:
                    flood(x - 1, y, cost + 1)
                if self.problem.map[x + 1][y].floor:
                    flood(x + 1, y, cost + 1)
                if self.problem.map[x][y - 1].floor:
                    flood(x, y - 1, cost + 1)
                if self.problem.map[x][y + 1].floor:
                    flood(x, y + 1, cost + 1)

        buff = [[float('inf') for _ in j] for j in self.problem.map]
        for target in self.problem.targets:
            visited = [[False for _ in i] for i in self.problem.map]
            flood(target[0], target[1], 0)

        return buff

    #count number of box moved
    def box_moved(self, current):
        count = 0
        for ind, val in enumerate(current):
            if val != self.box_state[ind]:
                count += 1
        self.box_state = current
        return count

    #improved heuristic function
    def heuristic2(self, s):
        box_pos = s.data[1:]
        if box_pos in self.memo:
            return self.memo[box_pos]
        targets = self.problem.targets
        matrix = self.problem.map
        box_moves = self.box_moved(box_pos)
        total = 0
        targets_left = len(targets)
        for val in box_pos:
            if val not in targets:
                if matrix[val[0] - 1][val[1]].wall and matrix[val[0]][val[1] - 1].wall:
                    self.memo[box_pos] = float('inf')
                    return float('inf')
                elif matrix[val[0] - 1][val[1]].wall and matrix[val[0]][val[1] + 1].wall:
                    self.memo[box_pos] = float('inf')
                    return float('inf')
                elif matrix[val[0] + 1][val[1]].wall and matrix[val[0]][val[1] - 1].wall:
                    self.memo[box_pos] = float('inf')
                    return float('inf')
                elif matrix[val[0] + 1][val[1]].wall and matrix[val[0]][val[1] + 1].wall:
                    self.memo[box_pos] = float('inf')
                    return float('inf')
            else:
                targets_left -= 1
            total += self.buff[val[0]][val[1]]
        self.memo[box_pos] = total * box_moves * targets_left
        return total * box_moves * targets_left

# solve sokoban map using specified algorithm
def solve_sokoban(map, algorithm='a2', dead_detection=False):
    # problem algorithm
    problem = SokobanProblem(map, dead_detection)
    # search algorithm
    h = Heuristic(problem).heuristic2 if ('2' in algorithm) else Heuristic(problem).heuristic
    if 'b' in algorithm:
        search = searchAlgo.bfs()
    else:
        search = searchAlgo.AStarSearch(heuristic=h)
    # solve problem
    search.solve(problem)
    return search.totalCost, search.actions, search.numStatesExplored

# animate the sequence of actions in sokoban map
def animate_sokoban_solution(map, seq, dt=0.2):
    print("-----Starting simulation-----")
    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'
    for i in range(len(seq)):
        os.system(clear)
        print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i + 1:])
        problem.print_state(state)
        time.sleep(dt)
        valid, _, state = problem.valid_move(state, seq[i])
        if not valid:
            raise Exception('Cannot move ' + seq[i] + ' in state ' + str(state))
    os.system(clear)
    print(seq)
    problem.print_state(state)


# read level map from file, returns map represented as string
def read_map_from_file(file, level):
    map = ''
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + level:
                    found = True
                    start = True
                    continue
            if start:
                if line[0] in SokobanProblem.valid_chars:
                    map += line
                else:
                    break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return map.strip('\n')


# extract all levels from file
def extract_levels(file):
    levels = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip().lower()[:5] == 'level':
                levels += [line.strip().lower()[6:]]
    return levels


def solve_map(file, level, algorithm, dead, simulate):
    map = read_map_from_file(file, level)
    print(map)
    tic = datetime.datetime.now()
    cost, sol, nstates = solve_sokoban(map, algorithm, dead)
    toc = datetime.datetime.now()
    print('Time consumed: {:.3f} seconds using {} and exploring {} states'.format(
        (toc - tic).seconds + (toc - tic).microseconds / 1e6, algorithm, nstates))
    process = psutil.Process(os.getpid())
    print("Memory used:", process.memory_info().rss / (1024 * 1024), "MB")
    if sol:
        seq = ''.join(sol)
        print("Solution:")
        print(len(seq), 'moves')
        print(' '.join(seq[i:i + 5] for i in range(0, len(seq), 5)))
        if simulate:
            animate_sokoban_solution(map, seq)
    else:
        print("dead dectection!")
        return


def main():
    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("level", help="Level name (1 => 15)")
    parser.add_argument("algorithm", help="a | a2 | b")
    parser.add_argument("-s", "--simulate", help="Simulate the solution (default off)", action="store_true")
    parser.add_argument("-f", "--file", help="File name storing the levels (levels.txt default)", default='levels.txt')

    args = parser.parse_args()
    level = args.level
    print('-----Starting solve level {} -----'.format(level))
    algorithm = args.algorithm
    dead = True
    simulate = args.simulate
    file = args.file

    #def solve_now():
    solve_map(file, level, algorithm, dead, simulate)

if __name__ == '__main__':
    main()