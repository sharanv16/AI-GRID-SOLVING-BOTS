from random import choice, random
from copy import deepcopy
from time import time
import csv

# cell constants
CLOSED_CELL = 1
TELEPORT_CELL = 2
OPEN_CELL = 4
CREW_CELL = 8
BOT_CELL = 16

# layout constantss
GRID_SIZE = 11
SUCCESS = 1
FAILURE = 0
CONV_ITERATIONS_LIMIT = 1000
CONVERGENCE_LIMIT = 1e-2 # Small value to reduce time complexity
TOTAL_ELEMENTS = 4

#Debugging
NO_CLOSED_CELLS = False
RAND_CLOSED_CELLS = 10
VISUALIZE = False

# moves constants
ALL_CREW_MOVES = [(1, 0), (0, 1), (-1, 0), (0, -1)]
ALL_BOT_MOVES = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

def get_manhattan_distance(cell_1, cell_2):
    return abs(cell_1[0] - cell_2[0]) + abs(cell_1[1] - cell_2[1])

class CELL:
    def __init__(self, i, j):
        self.state = OPEN_CELL
        self.no_bot_moves = float(GRID_SIZE**2)
        pos = (i, j)
        state = j + i*GRID_SIZE

class SHIP:
    def __init__(self, is_import = False):
        self.size = GRID_SIZE
        self.grid = [[CELL(i, j) for j in range(self.size)] for i in range(self.size)]
        self.open_cells = [ (j, i) for j in range(self.size) for i in range(self.size)]
        self.crew_pos = (0, 0)
        self.bot_pos = (0, 0)
        self.ideal_iters_limit = 0
        self.global_min_max = -(self.size**4*9*4)
        self.closed_cells = []
        self.wall_cells = []
        if not is_import:
            self.set_grid()
            self.place_players()

    def set_teleport(self):
        mid_point = (int)((self.size-1)/2)
        self.teleport_cell = (mid_point, mid_point)
        self.set_state(self.teleport_cell, TELEPORT_CELL)
        self.grid[mid_point][mid_point].no_bot_moves = 0

    def set_closed(self):
        mid_point = (int)((self.size-1)/2)
        other_points = [mid_point - 1, mid_point + 1]
        for i in other_points:
            for j in other_points:
                self.open_cells.remove((i, j))
                self.grid[i][j].no_bot_moves = 0
                self.set_state((i, j), CLOSED_CELL)
                self.closed_cells.append((i, j))

    def set_grid(self):
        self.set_teleport()
        if NO_CLOSED_CELLS:
            return

        self.set_closed()
        if RAND_CLOSED_CELLS:
            self.place_random_closed_cells()

    def place_random_closed_cells(self):
        random_closed = RAND_CLOSED_CELLS
        ignore_cells = [self.teleport_cell]
        for i in range(-2, 3):
            if i == 0:
                continue

            for (row,col) in [(0, 1*i),(1*i, 0)]:
                x_cord = self.teleport_cell[0] + row
                y_cord = self.teleport_cell[1] + col
                ignore_cells.append((x_cord, y_cord))

        while(True):
            random_cell = choice(self.open_cells)
            if random_cell not in ignore_cells:
                self.wall_cells.append(random_cell)
                random_closed -= 1
                self.set_state(random_cell, CLOSED_CELL)
                self.open_cells.remove(random_cell)

            if not random_closed:
                break

    def place_players(self):
        while(True):
            self.crew_pos = choice(self.open_cells)
            if self.crew_pos == self.teleport_cell or self.search_path(self.crew_pos):
                break

        crew_state = TELEPORT_CELL | CREW_CELL if self.get_state(self.crew_pos) & TELEPORT_CELL else CREW_CELL
        self.set_state(self.crew_pos, crew_state)
        self.open_cells.remove(self.crew_pos)

        while(True):
            self.bot_pos = choice(self.open_cells)
            if self.bot_pos == self.teleport_cell or self.search_path(self.bot_pos, False):
                break

        bot_state = TELEPORT_CELL | BOT_CELL if self.get_state(self.bot_pos) & TELEPORT_CELL else BOT_CELL
        self.set_state(self.bot_pos, bot_state)
        self.open_cells.remove(self.bot_pos)

    def set_state(self, pos, state_val):
        self.grid[pos[0]][pos[1]].state = state_val

    def set_moves(self, pos, moves):
        self.grid[pos[0]][pos[1]].no_bot_moves = moves

    def get_state(self, pos):
        return self.grid[pos[0]][pos[1]].state

    def get_cell(self, pos):
        return self.grid[pos[0]][pos[1]]

    def print_ship(self):
        for i in range(self.size):
            for j in range(self.size):
                # print(("%6s " % round(self.grid[i][j].no_bot_moves)), end=" ")
                print("%3s" % self.grid[i][j].state, end=" ")
            print()
        print("len ::", len(self.open_cells))

    def search_path(self, curr_pos, is_crew = True):
        bfs_queue = []
        visited_cells = set()
        bfs_queue.append((curr_pos, [curr_pos]))

        while bfs_queue:
            current_cell, path_traversed = bfs_queue.pop(0)
            if current_cell == self.teleport_cell:
                path_traversed.pop(0)
                return path_traversed
            elif (current_cell in visited_cells):
                continue

            visited_cells.add(current_cell)
            neighbors = self.get_all_moves(current_cell, ~CLOSED_CELL, is_crew)
            for neighbor in neighbors:
                if (neighbor not in visited_cells):
                    bfs_queue.append((neighbor, path_traversed + [neighbor]))

        return [] #God forbid, this should never happen

    def reset_grid(self):
        for cell in self.open_cells:
            self.set_state(cell, OPEN_CELL)

        self.set_state(self.teleport_cell, TELEPORT_CELL)
        crew_state = TELEPORT_CELL | CREW_CELL if self.get_state(self.crew_pos) & TELEPORT_CELL else CREW_CELL
        self.set_state(self.crew_pos, crew_state)
        bot_state = TELEPORT_CELL | BOT_CELL if self.get_state(self.bot_pos) & TELEPORT_CELL else BOT_CELL
        self.set_state(self.bot_pos, bot_state)

    def reset_positions(self):
        self.open_cells.append(self.bot_pos)
        self.open_cells.append(self.crew_pos)
        for cell in self.open_cells:
            self.set_state(cell, OPEN_CELL)

        self.set_state(self.teleport_cell, TELEPORT_CELL)
        self.place_players()

    def get_all_moves(self, curr_pos, filter = OPEN_CELL | TELEPORT_CELL, is_crew = True):
        neighbors = []

        all_moves = ALL_CREW_MOVES if is_crew else ALL_BOT_MOVES
        for (row,col) in all_moves:
            x_cord = curr_pos[0] + row
            y_cord = curr_pos[1] + col
            if (
                (0 <= x_cord < self.size)
                and (0 <= y_cord < self.size)
                and (self.get_state((x_cord, y_cord)) & filter)
            ):
                neighbors.append((x_cord, y_cord))

        return neighbors

    def perform_initial_calcs(self):
        # no bot
        self.calc_no_bot_steps()

        # bot - always calc no bot first, we can reuse those time steps instead of starting afresh
        self.set_state_details(2)
        self.set_calcs_lookup()
        self.perform_moves_iteration()
        self.perform_value_iteration()
        self.unset_global_states()

    def unset_global_states(self):
        del self.indi_states_lookup
        # del self.time_lookup
        del self.crew_moves
        del self.bot_moves
        del self.manhattan_lookup
        del self.bot_vs_crew_state

    def calc_no_bot_steps(self):
        total_iters = 0
        while(True):
            total_range = self.size**2
            for i in range(self.size):
                for j in range(self.size):
                    curr_cell = self.get_cell((i, j))
                    if curr_cell.state != CLOSED_CELL and (i, j) != self.teleport_cell:
                        neighbors = self.get_all_moves((i, j), ~(CLOSED_CELL))
                        moves_len = len(neighbors)
                        old_sum = curr_cell.no_bot_moves
                        if moves_len:
                            curr_cell.no_bot_moves = 1 + sum(self.get_cell(cell).no_bot_moves for cell in neighbors)/moves_len
                        else:
                            curr_cell.no_bot_moves = 0

                        if curr_cell.no_bot_moves - old_sum < CONVERGENCE_LIMIT:
                            total_range -= 1
                    else:
                        total_range -= 1

            total_iters += 1
            if total_range == 0 or total_iters >= CONV_ITERATIONS_LIMIT:
                self.ideal_iters_limit = 100 if total_iters < 100 else total_iters
                break

    def set_state_details(self, total_level):
        self.time_lookup = []
        self.indi_states_lookup = []
        for level in range(total_level):
            time_list = []
            indi_list = []
            for outer in range(self.size):
                if level == 0:
                    time_list.append([[self.get_cell((i, j)).no_bot_moves for j in range(self.size)] for i in range(self.size)])
                    indi_list.append([[0.0 for j in range(self.size)] for i in range(self.size)])
                else:
                    time_list.append(deepcopy(self.time_lookup))
                    indi_list.append(deepcopy(self.indi_states_lookup))

            self.time_lookup = time_list
            self.indi_states_lookup = indi_list

    def set_calcs_lookup(self):
        self.bot_vs_crew_state = dict()
        self.manhattan_lookup = dict()
        self.bot_moves = dict()
        self.crew_moves = dict()
        for row in range(self.size):
            for col in range(self.size):
                bot_pos = (row, col)
                if self.get_state(bot_pos) == CLOSED_CELL:
                    continue

                if bot_pos != self.teleport_cell and len(self.search_path(bot_pos, False)) == 0:
                    continue

                self.bot_vs_crew_state[bot_pos] = list()
                self.manhattan_lookup[bot_pos] = dict()
                bot_movements = self.get_all_moves(bot_pos, ~CLOSED_CELL, False)
                bot_movements.append(bot_pos)
                self.bot_moves[bot_pos] = bot_movements
                crew_states = self.bot_vs_crew_state[bot_pos]
                distances = self.manhattan_lookup[bot_pos]
                for i in range(self.size):
                    for j in range(self.size):
                        crew_pos = (i, j)
                        if self.get_state(crew_pos) == CLOSED_CELL or crew_pos == bot_pos:
                            continue

                        if crew_pos != self.teleport_cell and len(self.search_path(crew_pos)) == 0:
                            continue

                        crew_states.append(crew_pos)
                        distances[crew_pos] = get_manhattan_distance(crew_pos, bot_pos)
                        if crew_pos not in self.crew_moves:
                            self.crew_moves[crew_pos] = self.get_all_moves(crew_pos, ~CLOSED_CELL)

    def get_final_crew_moves(self, bot_pos, crew_pos):
        curr_bot_distance = self.manhattan_lookup[bot_pos][crew_pos]
        new_movements = []
        if curr_bot_distance == 1:
            escape_moves = []
            max_dist = 0
            for crew_move in self.crew_moves[crew_pos]:
                if crew_move == bot_pos:
                    continue

                new_dist =  self.manhattan_lookup[bot_pos][crew_move]
                escape_moves.append((crew_move, new_dist))
                if max_dist < new_dist:
                    max_dist = new_dist

            for escape in escape_moves:
                if escape[1] == max_dist:
                    new_movements.append(escape[0])
        else:
            new_movements = list(self.crew_moves[crew_pos])

        if not len(new_movements):
            new_movements.append(crew_pos)

        return new_movements

    def perform_moves_iteration(self):
        moves_possible = {}
        prev_difference = 0.0
        for iters in range(self.ideal_iters_limit):
            avg_difference = 0.0
            for bot_pos in self.bot_vs_crew_state:
                for crew_pos in self.bot_vs_crew_state[bot_pos]:
                    if crew_pos == self.teleport_cell:
                        continue

                    min_max = 1
                    for bot_action in self.bot_moves[bot_pos]:
                        if bot_action == crew_pos:
                            continue

                        crew_movements = self.get_final_crew_moves(bot_action, crew_pos)
                        crew_move_prob = 1/len(crew_movements)
                        action_time_step = 0.0
                        for crew_move in crew_movements:
                            if crew_move == self.teleport_cell:
                                continue

                            action_time_step += self.time_lookup[bot_action[0]][bot_action[1]][crew_move[0]][crew_move[1]]*crew_move_prob

                        if action_time_step > min_max:
                            min_max = action_time_step

                    old_max = self.time_lookup[bot_pos[0]][bot_pos[1]][crew_pos[0]][crew_pos[1]]
                    self.time_lookup[bot_pos[0]][bot_pos[1]][crew_pos[0]][crew_pos[1]] = min_max
                    avg_difference += min_max - old_max

            convergence = prev_difference - avg_difference
            if convergence >= 0 and convergence < CONVERGENCE_LIMIT:
                break

            prev_difference = avg_difference

    def perform_value_iteration(self):
        self.best_policy_lookup = {}
        max_iters = 0
        for iters in range(self.ideal_iters_limit):
            current_iters = 0
            for bot_pos in self.bot_vs_crew_state:
                for crew_pos in self.bot_vs_crew_state[bot_pos]:
                    if crew_pos == self.teleport_cell:
                        continue

                    min_max = self.global_min_max
                    action_val = ()
                    all_policies = {}
                    for bot_action in self.bot_moves[bot_pos]:
                        if bot_action == crew_pos:
                            continue

                        new_movements = self.get_final_crew_moves(bot_action, crew_pos)
                        crew_move_prob = 1/len(new_movements)
                        action_value = -1
                        for crew_move in new_movements:
                            if crew_move == self.teleport_cell:
                                continue

                            crew_reward = self.time_lookup[bot_action[0]][bot_action[1]][crew_move[0]][crew_move[1]]
                            new_state_value = self.indi_states_lookup[bot_action[0]][bot_action[1]][crew_move[0]][crew_move[1]]
                            action_value += (crew_move_prob*(new_state_value - crew_reward))

                        if action_value > min_max:
                            min_max = action_value
                            action_val = bot_action

                        all_policies[bot_action] = action_value

                    old_max = self.indi_states_lookup[bot_pos[0]][bot_pos[1]][crew_pos[0]][crew_pos[1]]
                    self.indi_states_lookup[bot_pos[0]][bot_pos[1]][crew_pos[0]][crew_pos[1]] = min_max
                    if iters == 0:
                        max_iters += 1
                    elif old_max - min_max < CONVERGENCE_LIMIT or (iters + 1) == self.ideal_iters_limit:
                        if bot_pos not in self.best_policy_lookup:
                            self.best_policy_lookup[bot_pos] = {}

                        bot_policy = self.best_policy_lookup[bot_pos]
                        bot_policy[crew_pos] = action_val
                        current_iters += 1

            if current_iters == max_iters:
                break

    def reset_static_pos(self, bot_pos):
        self.open_cells.append(self.bot_pos)
        self.open_cells.append(self.crew_pos)
        for cell in self.open_cells:
            self.set_state(cell, OPEN_CELL)

        self.set_state(self.teleport_cell, TELEPORT_CELL)

        self.bot_pos = bot_pos
        bot_state = TELEPORT_CELL | BOT_CELL if self.bot_pos == self.teleport_cell else BOT_CELL
        self.set_state(self.bot_pos, bot_state)
        self.open_cells.remove(self.bot_pos)

        while(True):
            self.crew_pos = choice(self.open_cells)
            if self.crew_pos == self.teleport_cell or self.search_path(self.crew_pos):
                break

        crew_state = TELEPORT_CELL | CREW_CELL if self.crew_pos == self.teleport_cell else CREW_CELL
        self.set_state(self.crew_pos, crew_state)
        self.open_cells.remove(self.crew_pos)


class PARENT_BOT:
    def __init__(self, ship):
        self.ship = ship
        self.local_crew_pos = self.ship.crew_pos
        self.local_bot_pos = self.ship.bot_pos
        self.csv_data = None
        self.local_all_moves = list(ALL_BOT_MOVES)
        self.local_all_moves.insert(0, (0,0))
        self.init_plots = False
        self.return_state = FAILURE
        self.best_move = (-1, -1)

    def move_bot(self):
        return

    def move_crew(self):
        return bool(True)

    def move_alien(self):
        return bool(False)

    def visualize_grid(self, is_end = True):
        if (not VISUALIZE) or (VISUALIZE and (not (self.ship.get_state(self.local_bot_pos) & BOT_CELL))):
            return

        data = []
        for i in range(self.ship.size):
            inner_list = []
            for j in range(self.ship.size):
                inner_list.append(self.ship.get_state((i, j)))

            data.append(inner_list)
        from matplotlib import pyplot
        if not self.init_plots:
            self.fig, self.ax = pyplot.subplots()
            self.image = pyplot.imshow(data, cmap='tab20')
            self.init_plots = True

        self.image.set_data(data)
        self.fig.canvas.draw_idle()
        pyplot.pause(.5)

        if is_end:
            pyplot.close(self.fig)

    def append_move(self):
        if self.csv_data is None:
            return

        self.csv_data.append([self.local_bot_pos, self.local_crew_pos])

    def start_rescue(self):
        self.total_moves = 0
        if self.ship.get_state(self.local_crew_pos) & TELEPORT_CELL:
            return self.total_moves, SUCCESS

        while(True):
            self.visualize_grid(False)
            self.total_moves += 1

            self.append_move()
            self.move_bot()
            if self.move_alien() or self.move_crew():
                self.append_move()
                self.visualize_grid()
                return self.total_moves, self.return_state

            if self.total_moves > 1000:
                self.append_move()
                self.visualize_grid()
                return self.total_moves, self.return_state

    def start_data_collection(self,filename):
        self.csv_data = []
        self.start_rescue()

        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for data in self.csv_data:
                writer.writerow(data)

        del self.csv_data
        return self.total_moves, self.return_state

class NO_BOT_CONFIG(PARENT_BOT):
    def __init__(self, ship):
        super(NO_BOT_CONFIG, self).__init__(ship)
        if self.ship.get_state(self.local_bot_pos) & TELEPORT_CELL:
            self.ship.set_state(self.local_bot_pos, TELEPORT_CELL)
        else:
            self.ship.set_state(self.local_bot_pos, OPEN_CELL)

    def move_crew(self):
        neighbors = self.ship.get_all_moves(self.local_crew_pos)
        if not neighbors:
            return False

        next_cell = choice(neighbors)
        self.ship.set_state(self.local_crew_pos, OPEN_CELL)
        old_pos = self.local_crew_pos
        self.local_crew_pos = next_cell
        next_state = CREW_CELL
        is_escaped = False
        if self.local_crew_pos == self.ship.teleport_cell:
            next_state |= TELEPORT_CELL
            self.return_state = SUCCESS
            is_escaped = True

        self.ship.set_state(next_cell, next_state)
        return is_escaped

class BOT_CONFIG(PARENT_BOT):
    def __init__(self, ship):
        super(BOT_CONFIG, self).__init__(ship)

    def make_bot_move(self, next_pos):
        old_pos = self.local_bot_pos
        old_state = TELEPORT_CELL if self.ship.get_state(old_pos) & TELEPORT_CELL else OPEN_CELL
        self.ship.set_state(self.local_bot_pos, old_state)
        self.local_bot_pos = next_pos
        next_state = BOT_CELL
        if self.ship.get_state(next_pos) & TELEPORT_CELL:
            next_state |= TELEPORT_CELL

        self.ship.set_state(next_pos, next_state)

    def move_bot(self):
        bot_movements = self.ship.get_all_moves(self.local_bot_pos, OPEN_CELL | TELEPORT_CELL, False)
        bot_movements.append(self.local_bot_pos)
        self.best_move = self.ship.best_policy_lookup[self.local_bot_pos][self.local_crew_pos]
        if not self.best_move:
            return

        self.make_bot_move(self.best_move)

    def get_next_move(self, neighbors):
        if self.local_bot_pos in neighbors:
            max_distance = 0
            new_neighbors = []
            for neighbor in neighbors:
                if neighbor != self.local_bot_pos:
                    bot_distance = get_manhattan_distance(neighbor, self.local_bot_pos)
                    new_neighbors.append((neighbor, bot_distance))
                    if max_distance < bot_distance:
                        max_distance = bot_distance

            neighbors.clear()
            for neighbor in new_neighbors:
                if max_distance == neighbor[1]:
                    neighbors.append(neighbor[0])

        if not neighbors:
            return None

        return choice(neighbors)

    def move_crew(self):
        neighbors = self.ship.get_all_moves(self.local_crew_pos)
        next_cell = self.get_next_move(neighbors)
        if next_cell is None:
            return False

        self.ship.set_state(self.local_crew_pos, OPEN_CELL)
        old_pos = self.local_crew_pos
        self.local_crew_pos = next_cell
        next_state = CREW_CELL
        is_escaped = False
        if self.local_crew_pos == self.ship.teleport_cell:
            next_state |= TELEPORT_CELL
            self.return_state = SUCCESS
            is_escaped = True

        self.ship.set_state(next_cell, next_state)
        return is_escaped
