from random import randint, uniform, choice
from math import e as exp, ceil, sqrt, floor
from inspect import currentframe
from time import time
from matplotlib import pyplot
from multiprocessing import Pool, cpu_count
from itertools import permutations
from heapq import heappop, heappush

#Constants
CLOSED_CELL = 1
OPEN_CELL = 2
BOT_CELL = 4
CREW_CELL = 8
ALIEN_CELL = 16
BOT_CAUGHT_CELL = 32
BOT_MOVEMENT_CELLS = OPEN_CELL | CREW_CELL | ALIEN_CELL
ALIEN_MOVEMENT_CELLS = CREW_CELL | OPEN_CELL | BOT_CELL
GRID_SIZE = 35
ADDITIVE_VALUE = 1e-6

BOT_SUCCESS = 1
BOT_FAILED = 2
BOT_STUCK = 3

X_COORDINATE_SHIFT = [1, 0, 0, -1]
Y_COORDINATE_SHIFT = [0, 1, -1, 0]

ALIEN_ZONE_SIZE = 5 # k - k >= 1, need to determine the large value
SEARCH_ZONE_SIZE = 5
ALPHA = 0.3 # avoid large alpha at the cost of performance

TOTAL_ITERATIONS = 1000
MAX_ALPHA_ITERATIONS = 5
MAX_K_ITERATIONS = 5
ALPHA_STEP_INCREASE = 0.1
ALIEN_ZONE_INCREASE = 1
TOTAL_BOTS = 8

DISTANCE_UTILITY=0.4 #DON'T WANT DISTANCE TO BE A MAJOR CONTRIBUTOR, SINCE EXPLORATION ALWAYS HELPS!??
ALIEN_UTILITY=-2.25 #ALIENS ARE ALWAYS DANGEROUS!?!
CREW_UTILITY=1.25 #GOAL IS TO SAVE THE CREW, SO LET'S GIVE IT SOME REWARD

LOG_NONE = 0
LOG_INFO = 1
LOG_DEBUG = 2
LOG_DEBUG_GRID = 3
IGNORE_GRID_DEBUG = True

LOOKUP_E = []
LOOKUP_NOT_E = []

ALIEN_NOT_PRESENT = 0.0
ALIEN_PRESENT = 1.0

ONE_ALIEN = 1
TWO_ALIENS = 2


# Common Methods
def get_neighbors(size, cell, grid, filter):
    neighbors = []

    for i in range(4):
        x_cord = cell[0] + X_COORDINATE_SHIFT[i]
        y_cord = cell[1] + Y_COORDINATE_SHIFT[i]
        if (
            (0 <= x_cord < size)
            and (0 <= y_cord < size)
            and (grid[x_cord][y_cord].cell_type & filter)
        ):
            neighbors.append((x_cord, y_cord))

    return neighbors

def get_manhattan_distance(cell_1, cell_2):
    return abs(cell_1[0] - cell_2[0]) + abs(cell_1[1] - cell_2[1])

def euclid_distance(point1, point2):
        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


""" simple logger to help with debugging """
class Logger:
    def __init__(self, log_level):
        self.log_level = log_level

    def check_log_level(self, log_level):
        return (self.log_level >= 0) and (log_level <= self.log_level)

    def print(self, log_level, *args):
        if self.check_log_level(log_level):
            print(currentframe().f_back.f_code.co_name, "::", currentframe().f_back.f_lineno, "::", sep="", end="")
            print(*args)

    def print_cell_data(self, log_level, crew_probs, cords, curr_pos):
        if self.check_log_level(log_level):
            print(currentframe().f_back.f_code.co_name, "::", currentframe().f_back.f_lineno, "::", sep="", end="")
            print(f"curr_pos::{curr_pos}, cell_cord::{(cords.cell_1, cords.cell_2) if type(cords) is Cell_Pair else cords}, cell_distance::{crew_probs.bot_distance}, crew_prob::{crew_probs.crew_prob}, beep_given_crew::{crew_probs.beep_given_crew}, no_beep_given_crew::{crew_probs.no_beep_given_crew}, crew_and_beep::{crew_probs.crew_and_beep}, crew_and_no_beep::{crew_probs.crew_and_no_beep}")

    def print_grid(self, grid, log_level = LOG_DEBUG_GRID):
        if not self.check_log_level(log_level):
            return

        print("****************")
        for i, cells in enumerate(grid):
            for j, cell in enumerate(cells):
                print("%10s" % (str(i) + str(j) + "::" + str(cell.cell_type)), end = " ")
            print("")
        print("****************")

    def print_crew_data(self, curr_pos, ship, crew_search_data, beep_count, is_beep_recv):
        print(currentframe().f_back.f_back.f_code.co_name, "::", currentframe().f_back.f_back.f_lineno, curr_pos, beep_count, is_beep_recv)
        print("%8s %27s %3s %27s %27s %27s %27s %27s %27s %27s %27s %20s" % ("cell", "p(C)", "B_D", "p(B|C)", "p(nB|c)", "p(C,B)", "p(C,nB)", "p(C|B)", "p(C|nB)", "p(B)",  "p(nB)",  "norm p(C)"))
        for cell_cord in crew_search_data.crew_cells:
            crew_probs = ship.get_cell(cell_cord).crew_probs
            print("%8s %27s %3s %27s %27s %27s %27s %27s %27s %27s %27s %20s" % (cell_cord, crew_probs.crew_prob, crew_probs.bot_distance, crew_probs.beep_given_crew, crew_probs.no_beep_given_crew, crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_given_beep, crew_probs.crew_given_no_beep, crew_search_data.beep_prob,  crew_search_data.no_beep_prob,  crew_search_data.normalize_probs))

    def print_all_crew_probs(self, curr_pos, ship, crew_search_data, total_beep, is_beep_recv):
        print(currentframe().f_back.f_back.f_code.co_name, "::", currentframe().f_back.f_back.f_lineno, curr_pos, total_beep, is_beep_recv)
        total_prob = 0.0
        for key in crew_search_data.crew_cells_pair:
            print(key, "::", end="\t")
            for cells in crew_search_data.crew_cells_pair[key]:
                total_prob += cells.crew_probs.crew_prob
                print(cells.cell_2.cord, "::", cells.crew_probs.crew_prob, end = " ")
            print()
        for key in crew_search_data.crew_cells_pair:
            print(key, "::", end="\t")
            for cells in crew_search_data.crew_cells_pair[key]:
                print(cells.cell_2.cord, "::", cells.crew_probs.no_beep_given_crew, end = " ")
            print()
        print(currentframe().f_back.f_back.f_code.co_name, "::", currentframe().f_back.f_back.f_lineno, curr_pos, total_prob, crew_search_data.crew_cells)

    def print_all_pair_data(self, log_level, bot):
        if not self.check_log_level(log_level):
            return

        self.print_all_crew_probs(bot.curr_pos, bot.ship, bot.crew_search_data, bot.total_beep, bot.crew_search_data.is_beep_recv)


    def print_all_crew_data(self, log_level, bot):
        if not self.check_log_level(log_level):
            return

        self.print_crew_data(bot.curr_pos, bot.ship, bot.crew_search_data, bot.total_beep, bot.crew_search_data.is_beep_recv)

    def print_map(self, cell, crew_cells_pair):
        if not self.check_log_level(LOG_DEBUG):
            return

        print(cell)
        for key in self.crew_cells_pair:
            print(str(key) + "::", end="")
            length = len(self.crew_cells_pair[key])
            for itr, val in enumerate(self.crew_cells_pair[key]):
                if itr < length - 1:
                    print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_prob) + ", ", end="")
                else:
                    print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_prob), end="")
            print()

    def print_heat_map(self, grid, is_beep_recv, curr_pos, is_crew = True):
        if IGNORE_GRID_DEBUG and not self.check_log_level(LOG_DEBUG_GRID):
          return

        prob_grid = []
        prob_spread = list()
        for cells in grid:
            prob_cell = []
            for cell in cells:
                probs = 0
                if is_crew:
                    probs = cell.crew_probs.crew_prob
                else:
                    probs = cell.alien_probs.alien_prob

                if cell.cell_type == CLOSED_CELL:
                    prob_cell.append(float('nan'))
                elif is_crew and cell.cell_type == (BOT_CELL|CREW_CELL):
                    prob_cell.append(1)
                else:
                    prob_cell.append(probs)
                    if not probs in prob_spread:
                        prob_spread.append(probs)
            prob_grid.append(prob_cell)

        prob_spread.sort()
        max_len = len(prob_spread) - 1

        if is_crew:
            prob_grid[curr_pos[0]][curr_pos[1]] = prob_spread[max_len]

        pyplot.figure(figsize=(10,10))
        pyplot.colorbar(pyplot.imshow(prob_grid, vmin=prob_spread[0], vmax=prob_spread[max_len]))
        pyplot.title("Beep recv" if is_beep_recv else "Beep not recv")
        pyplot.show()


# Modularizing our knowledge base for readability
class One_Alien_Evasion_Data:
    def __init__(self, ship):
        self.ship = ship
        self.init_alien_cells = list(self.ship.initial_alien_cells)
        self.alien_cells = list(self.ship.open_cells)
        self.alien_cells.extend([self.ship.crew_1, self.ship.crew_2, self.ship.bot])
        self.present_alien_cells = []
        self.alien_movement_cells = set()
        self.is_beep_recv = False
        self.beep_prob = 0.0
        self.no_beep_prob = 0.0
        self.beep_count = 0

        self.init_alien_cell_size = len(self.init_alien_cells)
        self.alien_cell_size = len(self.alien_cells)

    def init_alien_calcs(self, curr_pos):
        prob_alien_cells = self.ship.get_outer_cells(curr_pos) + self.ship.get_inner_border(curr_pos)
        prob_alien_cells_len = len(prob_alien_cells)
        for cell_cord in prob_alien_cells:
            cell = self.ship.get_cell(cell_cord)
            cell.alien_probs.alien_prob = 1/prob_alien_cells_len
            self.present_alien_cells.append(cell_cord)

class Two_Alien_Evasion_Data(One_Alien_Evasion_Data):
    def __init__(self, ship):
        self.alien_cells_pair = dict()
        self.visited_cells = set()
        super(Two_Alien_Evasion_Data, self).__init__(ship)

    def init_alien_calcs(self, curr_pos):
        prob_alien_cells = self.ship.get_outer_cells(curr_pos) + self.ship.get_inner_border(curr_pos)
        alien_cell_prob = len(prob_alien_cells) * (self.alien_cell_size - 1) / 2
        for key_itr, key in enumerate(self.alien_cells):
            if key_itr == self.alien_cell_size - 1:
                continue

            for val_itr in range(key_itr + 1, self.alien_cell_size):
                val = self.alien_cells[val_itr]
                key_val = (key, val)
                cell_pair = Alien_Cell_Pair(key, val, self.ship)
                cell_1 = cell_pair.cell_1
                cell_2 = cell_pair.cell_2
                if key not in self.alien_cells_pair:
                    self.alien_cells_pair[key] = dict()

                self.alien_cells_pair[key][val] = cell_pair
                cell_pair.alien_probs.alien_prob = 0
                if cell_1.cord in prob_alien_cells or cell_2.cord in prob_alien_cells:
                    cell_pair.alien_probs.alien_prob = 1/alien_cell_prob
                    self.present_alien_cells.append(key_val)

class Alien_Probs:
    def __init__(self):
        self.alien_prob = ALIEN_NOT_PRESENT
        self.alien_and_beep = ALIEN_NOT_PRESENT
        self.alien_and_no_beep = ALIEN_PRESENT

"""
    contains the probability related to a crew in each cell, i.e,
    bot_distance, the distance of the bot cell to this "probable" crew cell
    crew_prob, the "computed probability" of a crew in this cell
    beep_given_crew, the "probability" of hearing a beep from this cell given a crew is present
    crew_and_beep, the "probability" of the presence of both crew and beep to be in this cell
    crew_given_beep, the "computed probability" using Bayesian Networks
"""
class Crew_Probs:
    def __init__(self):
        self.bot_distance = 0 # distance of curr cell to bot
        self.crew_prob = 0.0 # p(c)
        self.beep_given_crew = 0.0 # p(b|c)
        self.no_beep_given_crew = 0.0 # p(¬b|c)
        self.crew_given_beep = 0.0 # p(c|b)
        self.crew_given_no_beep = 0.0 # p(c|¬b)
        self.crew_and_beep = 0.0 # p (c,b)
        self.crew_and_no_beep = 0.0 # p(c,¬b)

    def update_crew_probs(self, crew_search_data):
        self.crew_given_beep = (self.crew_and_beep) / crew_search_data.beep_prob
        if (crew_search_data.no_beep_prob != 0):
            self.crew_given_no_beep = (self.crew_and_no_beep) / crew_search_data.no_beep_prob

        self.crew_prob = self.crew_given_beep if crew_search_data.is_beep_recv else self.crew_given_no_beep
        self.crew_and_beep = self.beep_given_crew * self.crew_prob
        self.crew_and_no_beep = self.no_beep_given_crew * self.crew_prob
        crew_search_data.normalize_probs += self.crew_prob

class Beep:
    def __init__(self):
        self.crew_1_dist = self.crew_2_dist = 0 # distance of current cell to crew
        self.c1_beep_prob = self.c2_beep_prob = 0 # probability of hearing the crews beep from this cell

"""
    Since the grid is an array, each block of the array is a class of data
    This helps us maintain a lot of data related to the ship structure, and the probabilites of each cell
"""
class Cell: # contains the detail inside each cell (i, j)
    def __init__(self, row, col, cell_type = OPEN_CELL):
        self.cell_type = cell_type # static constant
        self.within_detection_zone = False # used to check if current cell is with-in detection zone of the alien
        self.crew_probs = Crew_Probs()
        self.alien_probs = Alien_Probs()
        self.listen_beep = Beep()
        self.cord = (row, col) # coordinate of this cell
        self.adj_cells = []
        self.zone_number = 0 # let the hunger games begin :p

"""
    Cell Pair is used when there are "two" crews or alien to be searched or looked out for
"""
class Cell_Pair:
    def __init__(self, cell_1, cell_2, ship):
        self.cell_1 = ship.get_cell(cell_1)
        self.cell_2 = ship.get_cell(cell_2)
        self.cells = [cell_1, cell_2]
        self.init_probs()

    def init_probs(self):
        self.crew_probs = Crew_Probs()

class Alien_Cell_Pair(Cell_Pair):
    def __init__(self, cell_1, cell_2, ship):
        super(Alien_Cell_Pair, self).__init__(cell_1, cell_2, ship)

    def init_probs(self):
        self.alien_probs = Alien_Probs()

"""
    Parent class containing crew search related data
"""
class Crew_Search_Data:
    def __init__(self):
        self.beep_prob = 0.0  # p(b) -> normalized for hearing beeps
        self.no_beep_prob = 0.0 # p(¬b) -> normalized for hearing no beeps
        self.normalize_probs = 0.0 # probs will be reduced from this to normalize them
        self.beep_count = 0
        self.is_beep_recv = False

    def set_all_probs(self, beep_prob = 0.0, no_beep_prob = 0.0, norm_prob = 0.0):
        self.beep_prob = beep_prob
        self.no_beep_prob = no_beep_prob
        self.normalize_probs = norm_prob

    def update_all_probs(self, beep_prob = 0.0, no_beep_prob = 0.0, norm_prob = 0.0):
        self.beep_prob += beep_prob
        self.no_beep_prob += no_beep_prob
        self.normalize_probs += norm_prob

"""
    Class for maintaing the list of possible crew cells and it's probability for 1 crew member
"""
class One_Crew_Search_DS(Crew_Search_Data):
    def __init__(self, ship):
        super(One_Crew_Search_DS, self).__init__()
        self.crew_cells = list(ship.open_cells) # list of all possible crew cells
        self.crew_cells.append(ship.crew_1)
        self.crew_cells.append(ship.crew_2)
        self.followup_list = list()
        self.all_crew_zones = {}
        for key in ship.open_cell_zones:
            self.all_crew_zones[key] = list(ship.open_cell_zones[key])

    def update_cell_mov_vals(self, crew_probs, curr_pos, cell_cord):
        crew_probs.bot_distance = get_manhattan_distance(curr_pos, cell_cord)
        crew_probs.beep_given_crew = LOOKUP_E[crew_probs.bot_distance]
        crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]
        crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
        crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
        self.beep_prob += crew_probs.crew_and_beep
        self.no_beep_prob += crew_probs.crew_and_no_beep

    def remove_cell_probs(self, rem_cell, curr_pos, logger):
        if rem_cell.cord not in self.crew_cells:
            return False

        if self.normalize_probs == 1:
            self.followup_list.clear()

        self.crew_cells.remove(rem_cell.cord)
        self.normalize_probs -= rem_cell.crew_probs.crew_prob
        self.followup_list.append((rem_cell.cord, rem_cell.crew_probs.crew_prob))
        logger.print(LOG_DEBUG, f"Removing cell {rem_cell.cord} from list of probable crew cells{self.crew_cells}")
        return True

    def init_crew_calcs(self, ship, curr_pos):
        crew_cell_size = len(self.crew_cells)
        self.set_all_probs()
        for cell_cord in self.crew_cells:
            cell = ship.get_cell(cell_cord)
            crew_probs = cell.crew_probs
            crew_probs.crew_prob = 1/crew_cell_size
            crew_probs.bot_distance = get_manhattan_distance(curr_pos, cell_cord)
            crew_probs.beep_given_crew = LOOKUP_E[crew_probs.bot_distance]
            crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]
            crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
            crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
            self.update_all_probs(crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_prob)

"""
    Class for maintaing the list of possible crew cells and it's probability for 2 crew members
"""
class Two_Crew_Search_DS(One_Crew_Search_DS):
    def __init__(self, ship, logger):
        super(Two_Crew_Search_DS, self).__init__(ship)
        self.crew_cells_pair = dict()
        self.logger = logger
        crew_cells_len = len(self.crew_cells)
        self.pending_crew_to_save = 0
        self.saved_crew_cell = ()
        self.crew_cells_length = crew_cells_len * (crew_cells_len - 1)

    def update_cell_mov_vals(self, crew_probs, curr_pos, cell_cord):
        crew_probs.bot_distance = get_manhattan_distance(curr_pos, cell_cord)
        crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]

    def update_crew_pair(self, cell_pair, curr_pos):
        cell_1 = cell_pair.cell_1.cord
        cell_2 = cell_pair.cell_2.cord
        if self.pending_crew_to_save == 1:
            self.update_cell_mov_vals(cell_pair.cell_1.crew_probs, curr_pos, cell_1)
            cell_pair.cell_2.crew_probs.bot_distance = 0
            cell_pair.cell_2.crew_probs.no_beep_given_crew = 1
        elif self.pending_crew_to_save == 2:
            cell_pair.cell_1.crew_probs.bot_distance = 0
            cell_pair.cell_1.crew_probs.no_beep_given_crew = 1
            self.update_cell_mov_vals(cell_pair.cell_2.crew_probs, curr_pos, cell_2)
        else:
            self.update_cell_mov_vals(cell_pair.cell_1.crew_probs, curr_pos, cell_1)
            self.update_cell_mov_vals(cell_pair.cell_2.crew_probs, curr_pos, cell_2)

    def update_cell_pair_vals(self, cell_pair, curr_pos):
        self.update_crew_pair(cell_pair, curr_pos)
        crew_probs = cell_pair.crew_probs
        crew_probs.no_beep_given_crew = cell_pair.cell_1.crew_probs.no_beep_given_crew * cell_pair.cell_2.crew_probs.no_beep_given_crew
        crew_probs.beep_given_crew = 1 - crew_probs.no_beep_given_crew
        crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
        crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
        self.beep_prob += crew_probs.crew_and_beep
        self.no_beep_prob += crew_probs.crew_and_no_beep
        self.normalize_probs += crew_probs.crew_prob

    def update_rem_crew_probs(self, crew_pair):
        self.normalize_probs -= crew_pair.crew_probs.crew_prob
        self.crew_cells_length -= 1
        del crew_pair

    # invoke when only 1 crew member is present
    def remove_cell_probs_1(self, rem_cell, curr_pos):
        index = self.crew_cells.index(rem_cell)
        if self.pending_crew_to_save == 1: # 1 is left, so remove key here
            cells_pair_list = self.crew_cells_pair[rem_cell]
            self.update_rem_crew_probs(cells_pair_list.pop(0))
            del self.crew_cells_pair[rem_cell]
        else: # 2 is left, so remove based on value here
            cell_pair_list = self.crew_cells_pair[self.saved_crew_cell]
            for itr, cell_pair in enumerate(cell_pair_list): # a tinsy bit costly here...
                if rem_cell == cell_pair.cell_2.cord:
                    self.update_rem_crew_probs(cell_pair)
                    cell_pair_list.pop(itr)
                    break

            if not len(cell_pair_list):
                del self.crew_cells_pair[self.saved_crew_cell]

    # invoke when 2 crew members are present
    def remove_cell_probs_2(self, rem_cell, curr_pos):
        index = self.crew_cells.index(rem_cell)
        for itr, key in enumerate(self.crew_cells):
            if itr < index:
                self.update_rem_crew_probs(self.crew_cells_pair[key].pop(index - 1))
            elif itr > index:
                self.update_rem_crew_probs(self.crew_cells_pair[key].pop(index))
            else:
                for crew_pair in self.crew_cells_pair[key]:
                    self.update_rem_crew_probs(crew_pair)

                del self.crew_cells_pair[key]

    def retain_success_cell_probs(self, curr_pos, pending_crew):
        if curr_pos not in self.crew_cells:
            return

        index = self.crew_cells.index(curr_pos)
        if pending_crew == 1: # retain with respect to value
            for itr, key in enumerate(self.crew_cells):
                final_index = index
                if itr < index:
                    final_index = index - 1
                elif itr == index:
                    cells_pair_list = self.crew_cells_pair[key]
                    for val_itr in range(len(self.crew_cells) - 1):
                        self.update_rem_crew_probs(cells_pair_list.pop(0))
                    del self.crew_cells_pair[key]
                    continue

                cells_pair_list = self.crew_cells_pair[key]
                for val_itr in range(len(self.crew_cells) - 1):
                    if val_itr < final_index:
                        self.update_rem_crew_probs(cells_pair_list.pop(0))
                    elif val_itr > final_index:
                        self.update_rem_crew_probs(cells_pair_list.pop(1))
        else: # retain with respect to key
            for itr, key in enumerate(self.crew_cells):
                cells_pair_list = self.crew_cells_pair[key]
                if itr != index:
                    for val_itr in range(len(self.crew_cells) - 1):
                        self.update_rem_crew_probs(cells_pair_list.pop(0))
                    del self.crew_cells_pair[key]
                else:
                    self.update_rem_crew_probs(cells_pair_list.pop(index))

        self.pending_crew_to_save = pending_crew
        self.saved_crew_cell = curr_pos
        self.logger.print(LOG_DEBUG, f"Retaining following cell {curr_pos} from the list of probable crew cells {self.crew_cells}")
        self.logger.print_map(curr_pos, self.crew_cells_pair)

    def init_crew_calcs(self, ship, curr_pos):
        self.set_all_probs()
        len_crew_cells = len(self.crew_cells)
        for key_itr, key in enumerate(self.crew_cells):
            self.crew_cells_pair[key] = list()
            cells_pair_list = self.crew_cells_pair[key]
            for val_itr, val in enumerate(self.crew_cells):
                if key_itr == val_itr:
                    continue
                cell_pair = Cell_Pair(key, val, ship)
                cell_pair.crew_probs.crew_prob = 1/self.crew_cells_length
                self.update_cell_pair_vals(cell_pair, curr_pos)
                cells_pair_list.append(cell_pair)

        self.logger.print_map(curr_pos, self.crew_cells_pair)


""" Core Ship Layout (same as last project) """
class Ship:
    def __init__(self, size, log_level = LOG_NONE):
        self.size = size
        self.grid = [[Cell(i, j, CLOSED_CELL) for j in range(size)] for i in range(size)] # grid is a list of list with each cell as a class
        self.open_cells = []
        self.logger = Logger(log_level)
        self.isBeep = 0
        self.bot = (0, 0)
        self.aliens = []
        self.initial_alien_pos = []
        self.crew_1 = (0, 0)
        self.crew_2 = (0, 0)
        self.open_cell_zones = {}

        self.generate_grid()

    def get_cell(self, cord):
        return self.grid[cord[0]][cord[1]]

    def generate_grid(self):
        self.assign_start_cell()
        self.unblock_closed_cells()
        self.unblock_dead_ends()
        self.compute_adj_cells()

    def assign_zones(self):
        region_size = int(self.size/SEARCH_ZONE_SIZE) - 1
        zone_limits = {}
        zones = 0
        for i in range(0, self.size, region_size + 1):
            for j in range(0, self.size, region_size + 1):
                zone_limit = [(i,j),(i+region_size,j),(i,j+region_size),(i+region_size,j+region_size)]
                zones += 1
                zone_limits[zones] = zone_limit

        for zone in zone_limits:
            regions_cords = zone_limits[zone]
            self.open_cell_zones[zone] = []
            for i in range(regions_cords[0][0], regions_cords[3][0] + 1):
                for j in range(regions_cords[0][1], regions_cords[3][1] + 1):
                    cell = self.get_cell((i, j))
                    if cell.cell_type != CLOSED_CELL and cell.cell_type != BOT_CELL:
                        self.open_cell_zones[zone].append((i, j))
                        cell.zone_number = zone

    def compute_adj_cells(self):
        for cell_cord in self.open_cells:
            cell = self.get_cell(cell_cord)
            neighbors = get_neighbors(
                self.size,
                cell_cord,
                self.grid,
                OPEN_CELL
            )

            cell.adj_cells = neighbors

    def assign_start_cell(self):
        random_row = randint(0, self.size - 1)
        random_col = randint(0, self.size - 1)
        self.grid[random_row][random_col].cell_type = OPEN_CELL
        self.open_cells.append((random_row, random_col))

    def unblock_closed_cells(self):
        available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)
        while len(available_cells):
            closed_cell = choice(available_cells)
            self.get_cell(closed_cell).cell_type = OPEN_CELL
            self.open_cells.append(closed_cell)
            available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)

    def unblock_dead_ends(self):
        dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)
        half_len = len(dead_end_cells)/2

        while half_len > 0:
            dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)
            half_len -= 1
            if len(dead_end_cells):
                continue
            dead_end_cell = choice(dead_end_cells)
            closed_neighbors = get_neighbors(
                self.size, dead_end_cell, self.grid, CLOSED_CELL
            )
            random_cell = choice(closed_neighbors)
            self.get_cell(random_cell).cell_type = OPEN_CELL
            self.open_cells.append(random_cell)

    def cells_with_one_open_neighbor(self, cell_type):
        results = []
        for row in range(self.size):
            for col in range(self.size):
                if ((self.grid[row][col].cell_type & cell_type) and
                    len(get_neighbors(
                        self.size, (row, col), self.grid, OPEN_CELL
                    )) == 1):
                    results.append((row, col))
        return results

    def set_player_cell_type(self):
        self.get_cell(self.bot).cell_type = BOT_CELL
        self.get_cell(self.crew_1).cell_type = CREW_CELL
        self.get_cell(self.crew_2).cell_type = CREW_CELL

    def place_crews(self):
        self.bot = choice(self.open_cells)
        self.open_cells.remove(self.bot)
        self.crew_1 = choice(self.open_cells)
        while (True):
            self.crew_2 = choice(self.open_cells)
            if self.crew_2 != self.crew_1:
                break

    def place_players(self):
        self.place_crews()
        self.open_cells.remove(self.crew_1)
        self.open_cells.remove(self.crew_2)
        for cell_cord in self.open_cells:
            self.init_cell_details(cell_cord)

        for cell in (self.crew_1, self.crew_2, self.bot):
            self.init_cell_details(cell)

        self.set_player_cell_type()
        self.assign_zones()
        self.reset_detection_zone()
        self.place_aliens(2)

    def init_cell_details(self, cell_cord):
        cell = self.get_cell(cell_cord)
        cell.listen_beep.crew_1_dist = get_manhattan_distance(cell_cord, self.crew_1)
        cell.listen_beep.crew_2_dist = get_manhattan_distance(cell_cord, self.crew_2)
        cell.listen_beep.c1_beep_prob = LOOKUP_E[cell.listen_beep.crew_1_dist]
        cell.listen_beep.c2_beep_prob = LOOKUP_E[cell.listen_beep.crew_2_dist]
        cell.cord = cell_cord

    def check_aliens_count(self, no_of_aliens):
        if len(self.aliens) == no_of_aliens:
            return

        alien_cell = self.aliens.pop(no_of_aliens)
        alien = self.get_cell(alien_cell)
        if alien.cell_type & CREW_CELL:
            self.get_cell(alien_cell).cell_type = CREW_CELL
        else:
            self.get_cell(alien_cell).cell_type = OPEN_CELL


    # The following methods are called from the bot
    def place_aliens(self, no_of_aliens):
        pending_aliens = no_of_aliens
        cells_within_zone = self.cells_within_bot_zone
        all_cells = list(self.open_cells)
        all_cells.extend([self.crew_1, self.crew_2])
        self.initial_alien_cells = [cell_cord for cell_cord in all_cells if cell_cord not in cells_within_zone]
        while(pending_aliens > 0):
            alien_cell = choice(self.initial_alien_cells)
            if alien_cell in self.initial_alien_pos: continue
            self.initial_alien_pos.append(alien_cell)

            if self.get_cell(alien_cell).cell_type & CREW_CELL:
                self.get_cell(alien_cell).cell_type |= ALIEN_CELL
            else:
                self.get_cell(alien_cell).cell_type = ALIEN_CELL

            pending_aliens -= 1

        self.aliens = list(self.initial_alien_pos)

    def move_aliens(self, bot):
        for itr, alien in enumerate(self.aliens):
            alien_cell = self.get_cell(alien)
            adj_cells = alien_cell.adj_cells
            alien_possible_moves = [adj_cell for adj_cell in adj_cells if self.get_cell(adj_cell).cell_type & ALIEN_MOVEMENT_CELLS]

            if len(alien_possible_moves) == 0:
                self.logger.print(
                    LOG_DEBUG,
                    f"Alien has no moves"
                )
                continue

            self.logger.print(
                LOG_DEBUG,
                f"Alien has moves {alien_possible_moves}"
            )

            alien_new_pos = choice(alien_possible_moves)
            old_alien_pos = alien_cell.cord
            self.aliens[itr] = alien_new_pos

            next_cell = self.get_cell(alien_new_pos)
            curr_cell = self.get_cell(old_alien_pos)

            if curr_cell.cell_type & CREW_CELL:
                curr_cell.cell_type = CREW_CELL
            else:
                curr_cell.cell_type = OPEN_CELL

            if next_cell.cell_type & BOT_CELL:
                self.logger.print(
                    LOG_INFO,
                    f"Alien moves from current cell {old_alien_pos} to bot cell {alien_new_pos}",
                )
                self.bot_caught_cell = alien_new_pos
                next_cell.cell_type = BOT_CAUGHT_CELL
                bot.is_caught = True
                return True

            else:
                self.logger.print(
                    LOG_DEBUG,
                    f"Alien moves from current cell {old_alien_pos} to open cell {alien_new_pos}",
                )
                if next_cell.cell_type & CREW_CELL:
                    next_cell.cell_type |= ALIEN_CELL
                else:
                    next_cell.cell_type = ALIEN_CELL

        return False

    def get_detection_zone(self, cell):
        k = ALIEN_ZONE_SIZE
        cells_within_zone = []
        min_row = max(0, cell[0] - k)
        max_row = min(self.size - 1, cell[0] + k)
        min_col = max(0, cell[1] - k)
        max_col = min(self.size - 1, cell[1] + k)
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell = self.get_cell((row, col))
                if cell.cell_type != CLOSED_CELL:
                    cells_within_zone.append((row, col))

        return cells_within_zone

    def get_outer_cells(self, curr_cell = None):
        if curr_cell is None:
            curr_cell = self.bot
        outer_cells = []
        k = ALIEN_ZONE_SIZE

        # Iterate over the cells surrounding the detection zone
        for i in range(curr_cell[0] - k - 1, curr_cell[0] + k + 2):
            for j in range(curr_cell[1] - k - 1, curr_cell[1] + k + 2):
                # Check if the cell is within the grid bounds and not part of the detection zone
                if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE \
                        and (i < curr_cell[0] - k or i > curr_cell[0] + k
                            or j < curr_cell[1] - k or j > curr_cell[1] + k):
                        cell = self.get_cell((i, j))
                        if cell.cell_type != CLOSED_CELL:
                             outer_cells.append((i, j))

        return outer_cells

    def get_inner_border(self, curr_cell = None):
        if curr_cell is None:
            curr_cell = self.bot
        inner_border = []
        k = ALIEN_ZONE_SIZE

        for i in range(curr_cell[0] - k, curr_cell[0] + k + 1):
            for j in range(curr_cell[1] - k, curr_cell[1] + k + 1):
                if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE\
                        and (i < curr_cell[0] - k + 1 or i > curr_cell[0] + k - 1
                        or j < curr_cell[1] - k + 1 or j > curr_cell[1] + k - 1):
                        cell = self.get_cell((i, j))
                        if cell.cell_type != CLOSED_CELL:
                            inner_border.append((i, j))

        return inner_border

    def reset_detection_zone(self, curr_cell = None):
        if curr_cell is None: curr_cell = self.bot
        self.cells_within_bot_zone = self.get_detection_zone(curr_cell)

        # Reset cells outside detection zone to false
        for i, cells in enumerate(self.grid):
            for j, cell in enumerate(cells):
                cell.within_detection_zone = ((i, j) in self.cells_within_bot_zone)

    def crew_beep_1(self, cell):
        self.isBeep = uniform(0, 1)
        return True if self.isBeep <= self.get_cell(cell).listen_beep.c1_beep_prob else False

    def crew_beep_2(self, cell):
        self.isBeep = uniform(0, 1)
        return True if self.isBeep <= self.get_cell(cell).listen_beep.c2_beep_prob else False

    def alien_beep(self):
        beep_heard = [alien_pos for alien_pos in self.aliens if self.get_cell(alien_pos).within_detection_zone]
        return len(beep_heard) > 0

    def reset_grid(self):
        # Need to make change here
        self.set_player_cell_type()
        self.reset_detection_zone()
        self.aliens = list(self.initial_alien_pos)
        for cell in self.open_cells:
            self.get_cell(cell).cell_type = OPEN_CELL


class CellSearchNode:
    def __init__(self, cord, parent=None):
        self.cord = cord
        self.parent = parent
        self.actual_est = 0
        self.total_cost = 0

    def __eq__(self, other):
        return self.cord == other.cord

    def __lt__(self, other):
        return self.total_cost < other.total_cost

""" Basic search algorithm, and the parent class for our bot """
class SearchAlgo:
    alien_config = ONE_ALIEN

    def __init__(self, ship, log_level):
        self.ship = ship
        self.curr_pos = ship.bot
        self.last_pos = ()
        self.logger = Logger(log_level)
        self.to_visit_list = []
        self.zone_vs_zone_prob = {}
        self.track_zones = {}
        # Working on few issues, will fix it ASAP
        self.disable_alien_calculation = True
        self.place_aliens_handler()

    def total_distance(self, points, path):
        total = 0
        for i in range(len(path) - 1):
            total += euclid_distance(points[path[i]], points[path[i + 1]])
        return total

    """
        Find the shortest route between multiple points, this ensures that when visiting from point A, B, C, D;
        point B and C are also visited along the way, instead of moving back and forth a lot.
    """
    def shortest_route(self, points):
        points.insert(0, self.curr_pos)
        n = len(points)
        shortest_distance = float('inf')
        shortest_path = None
        for path in permutations(range(n)):
            dist = self.total_distance(points, path)
            if dist < shortest_distance:
                shortest_distance = dist
                shortest_path = list(path)
        next_point_index = shortest_path[1]
        shortest_path_points = [points[i] for i in shortest_path]
        return next_point_index, shortest_path_points

    """
        A simple bfs based search algorithm that ignores the list of unsafe cell given to it.
    """
    def search_path(self, dest_cell, unsafe_cells = []):
        curr_pos = self.curr_pos

        bfs_queue = []
        visited_cells = set()
        bfs_queue.append((curr_pos, [curr_pos]))

        while bfs_queue:
            current_cell, path_traversed = bfs_queue.pop(0)
            if current_cell == dest_cell:
                path_traversed.pop(0)
                return path_traversed
            elif (current_cell in visited_cells):
                continue

            visited_cells.add(current_cell)
            neighbors = self.ship.get_cell(current_cell).adj_cells
            for neighbor in neighbors:
                if ((neighbor not in visited_cells) and
                    (neighbor not in unsafe_cells)):
                    bfs_queue.append((neighbor, path_traversed + [neighbor]))

        return [] #God forbid, this should never happen

    """
        A simple heuristic based search finding algorithm that ignores all unsafe cells.

        The cost is calculated as follows,
        Reduce cost based on prob of a crew in a cell (a reward, if i may)
        Increase cost based on prob of a alien in a cell (avoid at all costs??)
        Increase the cost 'slightly', based on distance (i don't think this is inherently bad because exploring more cells helps us with pruning unwanted data from our knowledge base)
    """
    def astar_search_path(self, dest_cell, unsafe_cells):
        curr_pos = self.curr_pos

        open_list = list()
        visited_set = set()
        start_node = CellSearchNode(curr_pos)
        goal_node = CellSearchNode(dest_cell)
        heappush(open_list, start_node)

        while open_list:
            current_node = heappop(open_list)

            if current_node == goal_node:
                path = []
                while current_node:
                    path.append(current_node.cord)
                    current_node = current_node.parent

                path = path[::-1]
                path.pop(0)
                return path

            visited_set.add(current_node.cord)

            for neighbour in self.ship.get_cell(current_node.cord).adj_cells:
                if (neighbour in visited_set) or (neighbour in unsafe_cells):
                    continue

                new_node = CellSearchNode(neighbour, current_node)

                cell = self.ship.get_cell(new_node.cord)
                alien_prob = cell.alien_probs.alien_prob
                crew_prob = cell.crew_probs.crew_prob
                new_node.actual_est = current_node.actual_est + 1
                new_node.total_cost = new_node.actual_est - (CREW_UTILITY * crew_prob) + (2.25 * alien_prob) # just avoid aliens!!!

                heappush(open_list, new_node)

        # No path found
        return []

    def place_aliens_handler(self):
        self.ship.check_aliens_count(self.alien_config)


""" Main parent class for all our bots, contain most of the common bot related logic """
class ParentBot(SearchAlgo):
    compute_movement_config = ONE_ALIEN

    def __init__(self, ship, log_level):
        super(ParentBot, self).__init__(ship, log_level)
        self.alien_evasion_data = One_Alien_Evasion_Data(ship) # to do only in one alien cases
        # self.alien_evasion_data = One_Alien_Evasion_Data(ship) # to do only in one alien cases
        self.temp_search_data = Crew_Search_Data()
        self.crew_search_data = One_Crew_Search_DS(ship)
        self.total_crew_to_save = 2
        self.traverse_path = []
        self.unsafe_cells = []
        self.bot_escape = False
        self.pred_crew_cells = list()
        self.is_caught = False
        self.is_own_design = self.is_bot_moved = self.is_escape_strategy = False
        self.recalc_pred_cells = True
        self.total_beep = self.track_beep = self.pending_crew = 0
        self.logger.print_grid(self.ship.grid)
        self.path_traversed = list()
        self.path_traversed.append(self.curr_pos)
        self.all_crews = [self.ship.crew_1, self.ship.crew_2]
        self.saved_crew = ()

    def rescue_info(self):
        init_1_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_1_dist
        self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_1}")
        init_2_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_2_dist
        self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_2}")
        return int(init_1_distance + self.ship.get_cell(self.ship.crew_1).listen_beep.crew_2_dist)

    def handle_single_crew_beep(self):
        self.crew_search_data.is_beep_recv = self.ship.crew_beep_1(self.curr_pos)
        self.logger.print(LOG_DEBUG, self.crew_search_data.is_beep_recv)

    def handle_two_crew_beep(self):
        beep_recv_2 = beep_recv_1 = False
        if self.pending_crew == 1:
            beep_recv_1 = self.ship.crew_beep_1(self.curr_pos)
        elif self.pending_crew == 2:
            beep_recv_2 = self.ship.crew_beep_2(self.curr_pos)
        else:
            beep_recv_1 = self.ship.crew_beep_1(self.curr_pos)
            beep_recv_2 = self.ship.crew_beep_2(self.curr_pos)

        self.crew_search_data.is_beep_recv = beep_recv_1 or beep_recv_2
        self.logger.print(LOG_DEBUG, beep_recv_1, beep_recv_2, beep_recv_1 or beep_recv_2)

    def handle_crew_beep(self):
        self.handle_single_crew_beep() if self.total_crew_to_save == 1 else self.handle_two_crew_beep()

        self.total_beep += 1
        if self.crew_search_data.is_beep_recv:
            self.track_beep += 1

    def handle_alien_beep(self):
        self.alien_evasion_data.is_beep_recv = self.ship.alien_beep()
        if self.alien_evasion_data.is_beep_recv:
            self.alien_evasion_data.beep_count += 1

    def calc_initial_search_data(self):
        self.crew_search_data.init_crew_calcs(self.ship, self.curr_pos)

    """
        instead of immediately escaping, we first try to find a path towards a cell most likely to contain the crew
        otherwise, we will escape towards a cell least likely to contain the alien
    """
    def find_escape_path(self):
        if self.to_visit_list:
            previous_visit = self.traverse_path[-1]
            for cell in self.to_visit_list:
                if self.is_own_design:
                    self.traverse_path = self.astar_search_path(cell, self.unsafe_cells)
                else:
                    self.traverse_path = self.search_path(cell, self.unsafe_cells)

                if self.traverse_path:
                    self.to_visit_list.remove(cell)
                    self.to_visit_list.append(previous_visit)
                    return self.traverse_path

            self.to_visit_list.append(previous_visit)

        safest_cells = []
        alien_present_cells = set()
        if self.compute_movement_config == ONE_ALIEN:
            alien_present_cells = self.alien_evasion_data.alien_movement_cells
        else:
            alien_present_cells = self.alien_evasion_data.visited_cells

        itr = 0
        for cell in self.alien_evasion_data.alien_cells:
            if itr == 5:
                break
            if cell not in alien_present_cells and self.ship.get_cell(cell).crew_probs.bot_distance <= 10:
                safest_cells.append(cell)
                itr += 1

        safest_cells = sorted(safest_cells, key=lambda cell:self.ship.get_cell(cell).crew_probs.bot_distance)
        for cell in safest_cells:
            if self.is_own_design:
                escape_path = self.astar_search_path(cell, self.unsafe_cells)
            else:
                escape_path = self.search_path(cell, self.unsafe_cells)

            if escape_path:
                return escape_path
        return []

    """
        If beep heard, cells within detection zone: P(B obs /A) = 1
        cells outside P(B obs /A) = 0
        If beep not heard, cells within detection zone: P(B obs /A) = 0
        cells outside P(B obs /A) = 1
        P(A/B obs) = P(B obs/ A) P(A) / P(B obs)

        having computed the likely alien movements, we update the prob of alien at each point here
    """
    def update_alien_data(self):
        is_additive_amoothing = False
        if not self.alien_evasion_data.beep_prob:
            self.alien_evasion_data.beep_prob = ADDITIVE_VALUE * len(self.alien_evasion_data.alien_movement_cells)

        for cell_cord in self.alien_evasion_data.alien_movement_cells:
            cell = self.ship.get_cell(cell_cord)
            if is_additive_amoothing:
                cell.alien_probs.alien_and_beep += ADDITIVE_VALUE
            cell.alien_probs.alien_prob = cell.alien_probs.alien_and_beep/self.alien_evasion_data.beep_prob
            if cell.zone_number not in self.zone_vs_zone_prob:
                self.zone_vs_zone_prob[cell.zone_number] = 0

            self.zone_vs_zone_prob[cell.zone_number] += ALIEN_UTILITY*cell.alien_probs.alien_prob

        self.unsafe_cells = sorted(self.alien_evasion_data.alien_movement_cells, key=lambda cell:self.ship.get_cell(cell).alien_probs.alien_prob, reverse=True)[:(ALIEN_ZONE_SIZE + self.alien_config)]
        unsafe_neighbors = []
        for cell_cord in self.unsafe_cells:
            cell = self.ship.get_cell(cell_cord)
            unsafe_neighbors.extend(cell.adj_cells)

        self.unsafe_cells.extend(unsafe_neighbors)

    """
        this will contain the list of cells that were previously detected to contain the aliens
        on the initial iteration, it will the cells right outside the detection zone and the cells right inside the detection zone
        this way, we can always account for the prob of the alien having moved in either from outside the detection zone,
        or alien moving further inside the detection zone due to the bot movement
    """
    def compute_likely_alien_movements(self):
        beep_recv = self.alien_evasion_data.is_beep_recv

        prob_cell_mapping = dict()
        for cell_cord in self.alien_evasion_data.present_alien_cells:
            self.logger.print(
                LOG_DEBUG, f"Iterating for ::{cell_cord}"
            )
            cell = self.ship.get_cell(cell_cord)
            possible_moves = cell.adj_cells
            total_moves = len(possible_moves)

            if cell.cord not in prob_cell_mapping:
                prob_cell_mapping[cell.cord] = cell.alien_probs.alien_prob
                cell.alien_probs.alien_prob = 0

            if (total_moves == 0):
                cell.alien_probs.alien_prob = prob_cell_mapping[cell.cord]
                continue

            self.logger.print(
                LOG_DEBUG, f"prob_cell_mapping::{prob_cell_mapping}"
            )

            self.logger.print(
                LOG_DEBUG, f"Neighbours for the current cell::{possible_moves}"
            )

            for alien_move in possible_moves:
                adj_cell = self.ship.get_cell(alien_move)
                if alien_move not in prob_cell_mapping.keys():
                    prob_cell_mapping[alien_move] = adj_cell.alien_probs.alien_prob
                    adj_cell.alien_probs.alien_prob = 0

                adj_cell.alien_probs.alien_prob += prob_cell_mapping[cell.cord]/total_moves
                if (beep_recv and adj_cell.within_detection_zone and adj_cell != self.curr_pos):
                    adj_cell.alien_probs.alien_and_beep = adj_cell.alien_probs.alien_prob
                    self.alien_evasion_data.alien_movement_cells.add(alien_move)
                elif (not beep_recv and not adj_cell.within_detection_zone):
                    adj_cell.alien_probs.alien_and_beep = adj_cell.alien_probs.alien_prob
                    self.alien_evasion_data.alien_movement_cells.add(alien_move)
                else:
                    adj_cell.alien_probs.alien_prob = adj_cell.alien_probs.alien_and_beep = 0

                self.alien_evasion_data.beep_prob += adj_cell.alien_probs.alien_and_beep

        prob_cell_mapping.clear()

    def move_bot(self):
        if not self.traverse_path:
            return False

        self.ship.get_cell(self.curr_pos).cell_type = OPEN_CELL
        self.last_pos = self.curr_pos
        self.curr_pos = self.traverse_path.pop(0)
        curr_cell = self.ship.get_cell(self.curr_pos)
        if (curr_cell.cell_type & ALIEN_CELL):    #  OOPS BYE BYE
            curr_cell.cell_type |= ALIEN_CELL
            self.logger.print(LOG_INFO, f"Bot caught!!!! @ cell::{self.curr_pos}")
            self.is_caught = True
        elif (curr_cell.cell_type & CREW_CELL):
            curr_cell.cell_type |= BOT_CELL
            self.logger.print(LOG_INFO, f"Yay bot found a crew!! @ cell::{self.curr_pos}")
        else:
            curr_cell.cell_type = BOT_CELL
            self.ship.reset_detection_zone(self.curr_pos)

        self.logger.print(LOG_DEBUG, f"Bot {self.last_pos} has moved to {self.curr_pos} trying to find {self.total_crew_to_save} crew pending")
        return True

    def is_rescued(self):
        for itr, crew in enumerate(self.all_crews):
            if crew == self.curr_pos:
                if itr == 0:
                    self.pending_crew = 2
                else:
                    self.pending_crew = 1
                self.logger.print_all_crew_data(LOG_DEBUG, self)
                self.all_crews.remove(self.curr_pos)
                self.saved_crew = self.curr_pos
                break

        if len(self.all_crews) == 0:
            return True

        return False

    """
        This will be extended by 1 & 2 crew logic respectively.

        This will remove the given cell from it's list of possible crew cells.
        By continuously pruning data, we can single out the cell most likely to contain the crew.
    """
    def remove_cell(self, rem_cell):
        return

    """
        This will be extended by 1 & 2 crew logic respectively.

        The prob normalization that each function perform are the same, but,
        the way they perform will be different due to the different data structures that are used.
    """
    def norm_probs(self):
        return

    def remove_nearby_cells(self):
        crew_search_data = self.crew_search_data
        neighbors = self.ship.get_cell(self.curr_pos).adj_cells
        for neighbor in neighbors:
            if neighbor not in crew_search_data.crew_cells:
                continue

            self.remove_cell(neighbor)

    """
        Pre-handling for updating crew search data:

        If we don't receive a beep, we can positively remove the nearby cells if they were not removed already
        When we prune data from our knowledge base, we need to normalize all the prob
        When the bot moves, the bot has to update the p(B|C) & p(¬B|C) for all cells, and recompute all the data again. This will also be handled here.
    """
    def update_crew_search_data(self):
        crew_search_data = self.crew_search_data
        if not crew_search_data.is_beep_recv: # no beep heard
            self.remove_nearby_cells() # remove nearby cells if they were not removed already

        if self.is_bot_moved or (crew_search_data.normalize_probs < 1): # normalize our probs
            self.norm_probs() # normalize the probs based on the removal, or to account for movement
            self.is_bot_moved = False

        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def rem_cell_for_crew_search(self, rem_cell): # if cell is getting removed, let that remove the cell from the zone as well
        zone_number = rem_cell.zone_number
        if zone_number in self.crew_search_data.all_crew_zones:
            self.crew_search_data.all_crew_zones[zone_number].remove(rem_cell.cord)
            if len(self.crew_search_data.all_crew_zones[zone_number]) == 0:
                del self.crew_search_data.all_crew_zones[zone_number]

        rem_cell.crew_probs.crew_prob = 0 # better to make this 0 and forget about this entirely
        self.crew_search_data.crew_cells.remove(rem_cell.cord)

    def update_cell_crew_probs(self, crew_probs):
        crew_probs.crew_given_beep = crew_probs.crew_and_beep/self.crew_search_data.beep_prob
        if self.crew_search_data.no_beep_prob: # it is quite possible to for this value to become "0" unlike our beep_prob
            crew_probs.crew_given_no_beep = crew_probs.crew_and_no_beep/self.crew_search_data.no_beep_prob

        crew_probs.crew_prob = crew_probs.crew_given_beep if self.crew_search_data.is_beep_recv else crew_probs.crew_given_no_beep
        crew_probs.crew_and_beep = crew_probs.beep_given_crew*crew_probs.crew_prob
        crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew*crew_probs.crew_prob
        self.temp_search_data.update_all_probs(crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_prob)

    def get_best_zone(self):
        zone_as_list = []
        self.track_zones.clear()
        curr_zone = self.ship.get_cell(self.curr_pos).zone_number
        curr_zone_pos = (curr_zone%SEARCH_ZONE_SIZE, floor(curr_zone/SEARCH_ZONE_SIZE))
        for key in self.zone_vs_zone_prob:
            # ignoring the distance factor for selecting a zone
            distance = abs(curr_zone_pos[0] - key%SEARCH_ZONE_SIZE) + abs(curr_zone_pos[1] - floor(key/SEARCH_ZONE_SIZE))
            # self.zone_vs_zone_prob[key] -= 0.05*distance #don't want distance to play a major factor...
            if key in self.crew_search_data.all_crew_zones:
                zone_as_list.append((key, self.zone_vs_zone_prob[key], distance))

        zone_as_list = sorted(zone_as_list, key=lambda data:(-data[1], data[2]))
        if not zone_as_list:
            return sorted(self.crew_search_data.crew_cells, key= lambda cell:(-self.ship.get_cell(cell).crew_probs.crew_prob, self.ship.get_cell(cell).crew_probs.bot_distance))[:3]

        max_size = 4 if self.total_crew_to_save == 2 else 3
        zone_1 = sorted(self.crew_search_data.all_crew_zones[zone_as_list[0][0]], key=lambda cell:(-self.ship.get_cell(cell).crew_probs.crew_prob, self.ship.get_cell(cell).crew_probs.bot_distance))[:max_size]
        self.track_zones[zone_as_list[0][0]] = (zone_1, self.zone_vs_zone_prob[zone_as_list[0][0]])
        return zone_1

    def get_most_prob_cell(self):
        next_best_crew_cells = sorted(self.crew_search_data.crew_cells, key= lambda cell:(self.ship.get_cell(cell).crew_probs.bot_distance, -self.ship.get_cell(cell).crew_probs.crew_prob))[:4]
        more_prob_cells = sorted(self.crew_search_data.crew_cells, key= lambda cell:(-self.ship.get_cell(cell).crew_probs.crew_prob, self.ship.get_cell(cell).crew_probs.bot_distance))[:3]
        next_best_crew_cells.extend(more_prob_cells)
        return next_best_crew_cells

    """
        Quarter way in the zone, if it realises the zone prob has reduced, it will start moving towards another zone.
        However, present zone will not be fully explored.
        An advantage we might over this is, the zone might now start getting alien beeps, and it's better to escape already...
    """
    def is_continue_traversing(self):
        if not self.is_own_design or len(self.to_visit_list) == 0 or len(self.track_zones) == 0:
            return False

        dest = self.to_visit_list[0]
        zone_number = self.ship.get_cell(dest).zone_number
        if zone_number not in self.track_zones:
            return False

        if not zone_number in self.zone_vs_zone_prob:
            for element in self.track_zones[zone_number][0]:
                if element in self.to_visit_list:
                    self.to_visit_list.remove(element)

            del self.track_zones[zone_number]
            return False

        if self.zone_vs_zone_prob[zone_number] > self.track_zones[zone_number][1]:
            total_elements = len(self.track_zones[zone_number][0])
            common_elements = []
            for x, y in zip(self.to_visit_list, self.track_zones[zone_number][0]):
                if x == y:
                    common_elements.append(x)

            if len(common_elements) < total_elements * .75:
                for element in common_elements:
                    self.to_visit_list.remove(element)

                del self.track_zones[zone_number]
                return False

        return True

    """
        Crew path finding, and escape path finding logic
    """
    def calculate_best_path(self):
        if self.traverse_path and (any(to_check in self.traverse_path for to_check in self.unsafe_cells)):  # if curr path is in an unsafe cell, escape that pathn imprompto!!
            self.traverse_path = self.find_escape_path()
            if self.traverse_path:
                return True

            return False # sit and pray...

        if self.traverse_path:
            return True

        self.is_continue_traversing()

        if len(self.to_visit_list) == 0:
            if self.is_own_design:
                dummy, self.to_visit_list = self.shortest_route(self.get_best_zone())
            else:
                dummy, self.to_visit_list = self.shortest_route(self.get_most_prob_cell())
            index = self.to_visit_list.index(self.curr_pos)
            self.to_visit_list.pop(index)
            if index > 0:
                self.to_visit_list.reverse()

        prob_crew_cell = self.to_visit_list.pop(0)
        if self.is_own_design:
            self.traverse_path = self.astar_search_path(prob_crew_cell, self.unsafe_cells)
        else:
            if len(self.unsafe_cells):
                self.traverse_path = self.search_path(prob_crew_cell, self.unsafe_cells)

        if self.traverse_path:
            return True

        if (not self.alien_evasion_data.is_beep_recv):
            self.traverse_path = self.search_path(prob_crew_cell)

        if not self.traverse_path:
            self.logger.print(LOG_DEBUG, f"Unable to find a path....")
            return False

        self.logger.print(LOG_DEBUG, f"New path to cell {prob_crew_cell} was found, {self.traverse_path}")
        return True

    def get_saved(self): # returns number of crews saved
        saved = self.total_crew_to_save - len(self.all_crews)
        if self.total_crew_to_save == 1 and len(self.all_crews) == 1:
            saved = 1
        elif self.total_crew_to_save == 1 and len(self.all_crews) == 2:
            saved = 0

        return saved

    def start_rescue(self): # where it all begins....
        total_moves = total_iter = 0
        bot_moved = keep_moving = False
        init_distance = self.rescue_info()
        self.calc_initial_search_data()
        self.logger.print_all_crew_data(LOG_DEBUG, self)

        # self.logger.print_heat_map(self.ship.grid, self.crew_search_data.is_beep_recv, self.curr_pos)
        # self.logger.print_heat_map(self.ship.grid, self.alien_evasion_data.is_beep_recv, self.curr_pos, False)
        while (True): # Keep trying till you find the crew
            if total_iter >= 1000:
                return init_distance, total_iter, total_moves, BOT_STUCK, self.get_saved()

            total_iter += 1
            self.handle_alien_beep()
            self.handle_crew_beep()

            """
                Computation for aliens does not begin until the first alien beep is heard, this helps us to reduce the total compuations
            """
            if self.alien_evasion_data.beep_count > 0:
                if self.alien_evasion_data.beep_count == 1 and self.alien_evasion_data.is_beep_recv: # No need to init more than once...
                    self.alien_evasion_data.init_alien_calcs(self.curr_pos) # Initial alien cell will be the cells right outside & inside the alien zone, i.e., 2k+2 & 2k+1

                self.alien_evasion_data.beep_prob = 0 # reset probs
                self.alien_evasion_data.alien_movement_cells = set()
                self.compute_likely_alien_movements()
                self.alien_evasion_data.present_alien_cells = list(self.alien_evasion_data.alien_movement_cells)

            self.zone_vs_zone_prob.clear()
            if self.alien_evasion_data.beep_count > 0:
                self.update_alien_data()

            self.update_crew_search_data()
            # self.logger.print_heat_map(self.ship.grid, self.crew_search_data.is_beep_recv, self.curr_pos)
            # self.logger.print_heat_map(self.ship.grid, self.alien_evasion_data.is_beep_recv, self.curr_pos, False)

            if self.calculate_best_path():
                if self.move_bot():
                    if self.is_rescued():
                        return init_distance, total_iter, total_moves, BOT_SUCCESS, self.get_saved()
                    elif self.is_caught:
                        return init_distance, total_iter, total_moves, BOT_FAILED, self.get_saved()

                    self.is_bot_moved = True
                    bot_moved = True
                    total_moves += 1
                    if self.curr_pos in self.crew_search_data.crew_cells:
                        self.remove_cell(self.curr_pos) # current cell can be ignored from all possible crew_cells
                else:
                    bot_moved = keep_moving = False

            if self.ship.move_aliens(self):
                return init_distance, total_iter, total_moves, BOT_FAILED, self.get_saved()

""" Bot 1 as per given specification """
class Bot_1(ParentBot):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_1, self).__init__(ship, log_level)
        self.override_ship_details()

    def rescue_info(self):
        if self.total_crew_to_save == 1:
            init_1_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_1_dist
            self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_1}")
            return int(init_1_distance)
        else:
            return super(Bot_1, self).rescue_info()

    def override_ship_details(self):
        self.total_crew_to_save = self.total_crew_count = self.pending_crew = 1
        cell = self.ship.get_cell(self.ship.crew_2)
        if cell.cell_type & ALIEN_CELL:
            self.ship.get_cell(self.ship.crew_2).cell_type = ALIEN_CELL
        else:
            self.ship.get_cell(self.ship.crew_2).cell_type = OPEN_CELL

    def remove_cell(self, rem_cell):
        crew_search_data = self.crew_search_data
        cell = self.ship.get_cell(rem_cell)
        crew_probs = cell.crew_probs
        crew_search_data.normalize_probs -= crew_probs.crew_prob
        self.rem_cell_for_crew_search(cell)
        self.logger.print(LOG_DEBUG, f"Removed {rem_cell} cell from possible crew cells {crew_search_data.crew_cells}")

    def norm_probs(self):
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()
        alpha_num = 0.0
        if crew_search_data.normalize_probs <= 0:
            self.logger.print_all_crew_data(LOG_DEBUG, self)
            self.logger.print(LOG_DEBUG, "Using additive smoothing...")
            alpha_num = ADDITIVE_VALUE
            crew_search_data.normalize_probs = alpha_num * len(crew_search_data.crew_cells)
        elif crew_search_data.normalize_probs > 1.5:
            self.logger.print_all_crew_data(LOG_NONE, self)
            print(self.__class__.__name__, "THERE WAS A MAJOR NEWS!!!!")
            exit(0)

        for cell_cord in crew_search_data.crew_cells:
            cell = self.ship.get_cell(cell_cord)
            crew_probs = cell.crew_probs
            if alpha_num:
                crew_probs.crew_prob = crew_probs.crew_prob + alpha_num

            crew_probs.crew_prob = crew_probs.crew_prob / crew_search_data.normalize_probs
            if (self.is_bot_moved):
                crew_probs.bot_distance = get_manhattan_distance(self.curr_pos, cell_cord)
                crew_probs.beep_given_crew = LOOKUP_E[crew_probs.bot_distance]
                crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]
            crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
            crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
            temp_search_data.update_all_probs(crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_prob)

        crew_search_data.set_all_probs(temp_search_data.beep_prob, temp_search_data.no_beep_prob, temp_search_data.normalize_probs)
        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def update_crew_search_data(self):
        super(Bot_1, self).update_crew_search_data()
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()

        for cell_cord in crew_search_data.crew_cells: # update probs for this round
            cell = self.ship.get_cell(cell_cord)
            crew_probs = cell.crew_probs
            self.update_cell_crew_probs(crew_probs)
            if cell.zone_number not in self.zone_vs_zone_prob:
                self.zone_vs_zone_prob[cell.zone_number] = 0

            if cell.zone_number in self.crew_search_data.all_crew_zones:
                self.zone_vs_zone_prob[cell.zone_number] += CREW_UTILITY*crew_probs.crew_prob

        crew_search_data.set_all_probs(temp_search_data.beep_prob, temp_search_data.no_beep_prob, temp_search_data.normalize_probs)
        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def is_rescued(self): # since this will only be used by 1, 2 and 3 (reason for if condition), this has been overriden
        if self.total_crew_count == 1:
            if self.curr_pos == self.ship.crew_1:
                self.logger.print(LOG_INFO, f"Bot has saved crew member at cell {self.ship.crew_1}")
                self.all_crews.remove(self.curr_pos)
                return True
            return False
        else:
            return super(Bot_1, self).is_rescued()

class Bot_2(Bot_1):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_2, self).__init__(ship, log_level)
        self.is_own_design = True

class Bot_3(Bot_1):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_3, self).__init__(ship, log_level)
        self.override_ship_details()

    def is_rescued(self):
        old_val = self.pending_crew
        ret_val = super(Bot_3, self).is_rescued()

        """
            Upon rescuing the first crew member, we become aware of the presence of the second crew.
            It is highly likely that the data we have is inaccurate, so we reinitialize all the data and start all over again.
        """
        if old_val != self.pending_crew:
            self.calc_initial_search_data()
            self.is_bot_moved = False # recalc already, need not do it again...

        return ret_val

    def override_ship_details(self):
        self.total_crew_to_save = self.total_crew_count = 2
        self.pending_crew = 0
        cell = self.ship.get_cell(self.ship.crew_2)
        if cell.cell_type & ALIEN_CELL:
            self.ship.get_cell(self.ship.crew_2).cell_type |= CREW_CELL
        else:
            self.ship.get_cell(self.ship.crew_2).cell_type = CREW_CELL

class Bot_4(ParentBot):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_4, self).__init__(ship, log_level)
        self.crew_search_data = Two_Crew_Search_DS(ship, self.logger)

    def remove_cell(self, rem_cell):
        crew_search_data = self.crew_search_data
        if self.pending_crew: # when only one crew is pending, we remove crew differently
            crew_search_data.remove_cell_probs_1(rem_cell, self.curr_pos)
        else: # when only both crews are present, gotta remove them all
            crew_search_data.remove_cell_probs_2(rem_cell, self.curr_pos)

        cell = self.ship.get_cell(rem_cell)
        self.rem_cell_for_crew_search(cell)

    def norm_probs(self):
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()
        alpha_num = 0.0
        if crew_search_data.normalize_probs <= 0:
            self.logger.print_map(self.curr_pos, self.crew_search_data.crew_cells_pair)
            self.logger.print(LOG_DEBUG, "Using additive smoothing...")
            alpha_num = ADDITIVE_VALUE
            crew_search_data.normalize_probs = alpha_num * crew_search_data.crew_cells_length
        elif crew_search_data.normalize_probs > 1.5:
            self.logger.print_all_pair_data(LOG_NONE, self)
            print(self.__class__.__name__, self.pending_crew, self.ship.crew_1, self.ship.crew_2, self.all_crews, "THERE WAS A MAJOR NEWS!!!!")
            exit(0)

        for key in self.crew_search_data.crew_cells_pair:
            cell_pair_list = self.crew_search_data.crew_cells_pair[key]
            for cell_pair in cell_pair_list:
                crew_probs = cell_pair.crew_probs
                if alpha_num:
                    crew_probs.crew_prob = crew_probs.crew_prob + alpha_num

                crew_probs.crew_prob = crew_probs.crew_prob / crew_search_data.normalize_probs
                if (self.is_bot_moved):
                    crew_search_data.update_crew_pair(cell_pair, self.curr_pos)
                    crew_probs.no_beep_given_crew = cell_pair.cell_1.crew_probs.no_beep_given_crew * cell_pair.cell_2.crew_probs.no_beep_given_crew
                    crew_probs.beep_given_crew = 1 - crew_probs.no_beep_given_crew

                crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
                crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
                temp_search_data.update_all_probs(crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_prob)

        crew_search_data.set_all_probs(temp_search_data.beep_prob, temp_search_data.no_beep_prob, temp_search_data.normalize_probs)
        self.logger.print_all_pair_data(LOG_DEBUG, self)

    def update_crew_search_data(self):
        super(Bot_4, self).update_crew_search_data()
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()
        is_visited = {}

        """
            crew_cells_len can be one after saving a crew, if not,
            the number of time a prob is calc for a cell will be 2*n times, n being the size of the grid
        """
        crew_cells_len = 1 if self.pending_crew else (2 * len(self.crew_search_data.crew_cells))
        for key in crew_search_data.crew_cells_pair:
            for cell_pairs in crew_search_data.crew_cells_pair[key]:
                crew_probs = cell_pairs.crew_probs
                self.update_cell_crew_probs(crew_probs)
                cell_1 = cell_pairs.cell_1
                cell_2 = cell_pairs.cell_2

                if cell_1.cord not in is_visited:
                    is_visited[cell_1.cord] = 1
                    cell_1.crew_probs.crew_prob = 0.0

                if cell_2.cord not in is_visited:
                    is_visited[cell_2.cord] = 1
                    cell_2.crew_probs.crew_prob = 0.0

                if cell_1.cord != self.saved_crew:
                    if cell_1.zone_number not in self.zone_vs_zone_prob:
                        self.zone_vs_zone_prob[cell_1.zone_number] = 0.0

                    cell_1.crew_probs.crew_prob += crew_probs.crew_prob

                    if cell_1.zone_number in self.crew_search_data.all_crew_zones: #double checking to be safe...
                        self.zone_vs_zone_prob[cell_1.zone_number] += CREW_UTILITY*crew_probs.crew_prob/crew_cells_len

                if cell_2.cord != self.saved_crew:
                    if cell_2.zone_number not in self.zone_vs_zone_prob:
                        self.zone_vs_zone_prob[cell_2.zone_number] = 0.0

                    cell_2.crew_probs.crew_prob += crew_probs.crew_prob

                    if cell_2.zone_number in self.crew_search_data.all_crew_zones:
                        self.zone_vs_zone_prob[cell_2.zone_number] += CREW_UTILITY*crew_probs.crew_prob/crew_cells_len

        crew_search_data.set_all_probs(temp_search_data.beep_prob, temp_search_data.no_beep_prob, temp_search_data.normalize_probs)
        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def is_rescued(self):
        old_val = self.pending_crew
        ret_val = super(Bot_4, self).is_rescued()
        if old_val != self.pending_crew and len(self.all_crews):
            self.crew_search_data.retain_success_cell_probs(self.curr_pos, self.pending_crew)
            self.rem_cell_for_crew_search(self.ship.get_cell(self.curr_pos))
        return ret_val

class Bot_5(Bot_4):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_5, self).__init__(ship, log_level)
        self.is_own_design = True

class Bot_6(Bot_3):
    alien_config = TWO_ALIENS

    def __init__(self, ship, log_level=LOG_NONE):
        super(Bot_6, self).__init__(ship, log_level)

class Bot_7(Bot_4):
    alien_config = TWO_ALIENS
    compute_movement_config = TWO_ALIENS

    def __init__(self, ship, log_level=LOG_NONE):
        super(Bot_7, self).__init__(ship, log_level)
        self.alien_evasion_data = Two_Alien_Evasion_Data(self.ship)

    def compute_likely_alien_movements(self):
        beep_recv = self.alien_evasion_data.is_beep_recv
        self.alien_evasion_data.visited_cells = set()

        prob_cell_pair_mapping = dict()

        """
            this will contain the list of cells that were previously detected to contain the aliens
            on the initial iteration, it will the cells right outside the detection zone and the cells right inside the detection zone
            this way, we can always account for the prob of the alien having moved in either from outside the detection zone,
            or alien moving further inside the detection zone due to the bot movement

            key and value are the pair of keys likely to contain the alien
        """
        for key_val_cells in self.alien_evasion_data.present_alien_cells:
            possible_moves = dict()
            cell_pair = self.alien_evasion_data.alien_cells_pair[key_val_cells[0]][key_val_cells[1]]
            no_moves = 0
            cell_1 = cell_pair.cell_1
            cell_2 = cell_pair.cell_2

            total_moves = 1
            for cell in [cell_1, cell_2]:
                if len(cell.adj_cells) == 0:
                    no_moves += 1
                    continue
                possible_moves[cell.cord] = cell.adj_cells
                total_moves *= len(possible_moves[cell.cord])

            if ((no_moves == 2)):
                continue

            if key_val_cells[0] not in prob_cell_pair_mapping:
                prob_cell_pair_mapping[key_val_cells[0]] = dict()

            if key_val_cells[1] not in prob_cell_pair_mapping[key_val_cells[0]]:
                prob_cell_pair_mapping[key_val_cells[0]][key_val_cells[1]] = cell_pair.alien_probs.alien_prob
                cell_pair.alien_probs.alien_prob = 0

            # self.logger.print(
            #     LOG_DEBUG, f"prob_cell_mapping::{prob_cell_pair_mapping}"
            # )

            # self.logger.print(
            #     LOG_DEBUG, f"Neighbours for the current cell pair::{possible_moves}"
            # )

            # Cell pair movement logic
            for alien_moves_1 in possible_moves[cell_1.cord]:
                for alien_moves_2 in possible_moves[cell_2.cord]:
                    if alien_moves_1 == alien_moves_2:
                        continue

                    adj_key_val_cell = (alien_moves_1, alien_moves_2)
                    if adj_key_val_cell[0] not in self.alien_evasion_data.alien_cells_pair:
                        adj_key_val_cell = (alien_moves_2, alien_moves_1)
                        if adj_key_val_cell[0] not in self.alien_evasion_data.alien_cells_pair:
                            continue
                        if adj_key_val_cell[1] not in self.alien_evasion_data.alien_cells_pair[adj_key_val_cell[0]]:
                            continue

                    if adj_key_val_cell[1] not in self.alien_evasion_data.alien_cells_pair[adj_key_val_cell[0]]:
                        adj_key_val_cell = (alien_moves_2, alien_moves_1)
                        if adj_key_val_cell[0] not in self.alien_evasion_data.alien_cells_pair:
                            continue
                        if adj_key_val_cell[1] not in self.alien_evasion_data.alien_cells_pair[adj_key_val_cell[0]]:
                            continue

                    adj_cell_pair = self.alien_evasion_data.alien_cells_pair[adj_key_val_cell[0]][adj_key_val_cell[1]]
                    if adj_key_val_cell[0] not in prob_cell_pair_mapping:
                        prob_cell_pair_mapping[adj_key_val_cell[0]] = dict()

                    if adj_key_val_cell[1] not in prob_cell_pair_mapping[adj_key_val_cell[0]]:
                        prob_cell_pair_mapping[adj_key_val_cell[0]][adj_key_val_cell[1]] = cell_pair.alien_probs.alien_prob
                        cell_pair.alien_probs.alien_prob = 0

                    adj_cell_pair.alien_probs.alien_prob += prob_cell_pair_mapping[key_val_cells[0]][key_val_cells[1]]/total_moves
                    alien_cell_1 = self.ship.get_cell(alien_moves_1)
                    alien_cell_2 = self.ship.get_cell(alien_moves_2)
                    check_detetction = (alien_cell_1.within_detection_zone) or (alien_cell_1.within_detection_zone)
                    if beep_recv and check_detetction and (not (alien_moves_1 == self.curr_pos or alien_moves_2 == self.curr_pos)):
                        adj_cell_pair.alien_probs.alien_and_beep = adj_cell_pair.alien_probs.alien_prob
                    elif (not beep_recv) and (not check_detetction):
                        adj_cell_pair.alien_probs.alien_and_beep = adj_cell_pair.alien_probs.alien_prob
                    else:
                        alien_cell_1.alien_probs.alien_prob = 0.0
                        alien_cell_2.alien_probs.alien_prob = 0.0
                        adj_cell_pair.alien_probs.alien_prob = adj_cell_pair.alien_probs.alien_and_beep = 0

                    if adj_cell_pair.alien_probs.alien_prob:
                        self.alien_evasion_data.alien_movement_cells.add(adj_key_val_cell)
                        self.alien_evasion_data.visited_cells.add(alien_moves_1)
                        self.alien_evasion_data.visited_cells.add(alien_moves_2)
                        self.alien_evasion_data.beep_prob += adj_cell_pair.alien_probs.alien_and_beep

            possible_moves.clear()

        prob_cell_pair_mapping.clear()

    """
        having computed the likely alien movements, we update the prob of alien at each point here
    """
    def update_alien_data(self):
        is_additive_amoothing = False
        if not self.alien_evasion_data.beep_prob:
            self.alien_evasion_data.beep_prob = ADDITIVE_VALUE * len(self.alien_evasion_data.alien_movement_cells)

        # Updating the alien prob from prior knowledge
        total_cells = len(self.alien_evasion_data.visited_cells)
        for cell_pair_key in self.alien_evasion_data.alien_movement_cells:
            cell_pair = self.alien_evasion_data.alien_cells_pair[cell_pair_key[0]][cell_pair_key[1]]
            cell_pair.alien_probs.alien_prob = cell_pair.alien_probs.alien_and_beep/self.alien_evasion_data.beep_prob
            for cell_cord in cell_pair.cells:
                prob = cell_pair.alien_probs.alien_prob/total_cells
                cell = self.ship.get_cell(cell_cord)

                if cell.zone_number not in self.zone_vs_zone_prob:
                    self.zone_vs_zone_prob[cell.zone_number] = 0

                cell.alien_probs.alien_prob += prob
                self.zone_vs_zone_prob[cell.zone_number] += ALIEN_UTILITY*prob


        self.unsafe_cells = sorted(self.alien_evasion_data.visited_cells, key=lambda cell:self.ship.get_cell(cell).alien_probs.alien_prob, reverse=True)[:(ALIEN_ZONE_SIZE + self.alien_config)]
        unsafe_neighbors = []
        for cell_cord in self.unsafe_cells:
            cell = self.ship.get_cell(cell_cord)
            unsafe_neighbors.extend(cell.adj_cells)

        self.unsafe_cells.extend(unsafe_neighbors)

class Bot_8(Bot_7):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_8, self).__init__(ship, log_level)
        self.is_own_design = True

class Bonus_Bot(ParentBot):
    alien_config = ONE_ALIEN

    def __init__(self, ship, log_level = LOG_NONE):
        super(Bonus_Bot, self).__init__(ship, log_level)
        self.curr_crew = self.ship.crew_1
        self.total_crew_to_save = 1
        self.is_own_design = True
        self.remove_additional_crew()

    def remove_additional_crew(self):
        if self.ship.get_cell(self.all_crews[1]).cell_type & ALIEN_CELL:
            self.ship.get_cell(self.all_crews[1]).cell_type = ALIEN_CELL
        else:
            self.ship.get_cell(self.all_crews[1]).cell_type = OPEN_CELL
        self.all_crews.pop(1)

    def rescue_info(self):
        init_1_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_1_dist
        self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_1}")
        return int(init_1_distance)

    def move_crew(self):
        crew_cell = self.ship.get_cell(self.curr_crew)
        new_crew_cord = choice(crew_cell.adj_cells)
        if crew_cell.cell_type & ALIEN_CELL:
            crew_cell.cell_type = ALIEN_CELL
        else:
            crew_cell.cell_type = OPEN_CELL

        new_crew_cell = self.ship.get_cell(new_crew_cord)
        self.curr_crew = new_crew_cord
        curr_bot_cell = self.ship.get_cell(self.curr_pos)
        if new_crew_cell.cell_type & BOT_CELL:
            self.logger.print(LOG_INFO, f"Crew has moved from current cell {crew_cell.cord} to bot cell {self.curr_crew}")
            new_crew_cell.cell_type |= CREW_CELL
            return True
        elif new_crew_cell.cell_type & ALIEN_CELL:
            new_crew_cell.cell_type |= CREW_CELL
        else:
            new_crew_cell.cell_type = OPEN_CELL

        curr_bot_cell.listen_beep.crew_1_dist = get_manhattan_distance(self.curr_pos, self.curr_crew)
        curr_bot_cell.listen_beep.c1_beep_prob = LOOKUP_E[curr_bot_cell.listen_beep.crew_1_dist]
        self.logger.print(LOG_DEBUG, f"Crew has moved from cell {crew_cell.cord} to cell {self.curr_crew}")
        return False

    """
        This code works similar to the alien movement calculation logic,
        the only distinction being over here we consider receiving no beep is equal to some value,
        and that the cells next to us not containing a crew member in that case

        Apart from calculating the new probability for each cell, the normalization when the
        bot moves to a new position, or when a crew member is not present in an adjacent cell is also taken into account over here.
    """
    def compute_likely_bot_movements(self):
        self.crew_search_data.set_all_probs(0, 0, self.crew_search_data.normalize_probs)
        beep_recv = self.crew_search_data.is_beep_recv
        curr_pos_adj_cells = self.ship.get_cell(self.curr_pos).adj_cells
        prob_cell_mapping = dict()
        updated_cells = set()
        use_smoothing = False
        if not self.crew_search_data.normalize_probs:
            use_smoothing = True
            self.crew_search_data.normalize_probs = len(self.crew_search_data.crew_cells)*ADDITIVE_VALUE

        if not beep_recv:
            for cell_cord in curr_pos_adj_cells:
                if cell_cord in self.crew_search_data.crew_cells:
                    self.remove_cell(cell_cord)

        for cell_cord in self.crew_search_data.crew_cells:
            self.logger.print(
                LOG_DEBUG, f"Iterating for ::{cell_cord}"
            )
            cell = self.ship.get_cell(cell_cord)
            possible_moves = list(cell.adj_cells)
            if not beep_recv:
                for move in curr_pos_adj_cells:
                    if move in possible_moves:
                        possible_moves.remove(move)

            total_moves = len(possible_moves)

            if cell.cord not in prob_cell_mapping:
                if use_smoothing:
                    cell.crew_probs.crew_prob += ADDITIVE_VALUE
                prob_cell_mapping[cell.cord] = cell.crew_probs.crew_prob/self.crew_search_data.normalize_probs
                cell.crew_probs.crew_prob = 0

            if (total_moves == 0):
                cell.crew_probs.crew_prob = prob_cell_mapping[cell.cord]
                continue

            for crew_move in possible_moves:
                adj_cell = self.ship.get_cell(crew_move)
                if crew_move not in prob_cell_mapping.keys():
                    if use_smoothing:
                        cell.crew_probs.crew_prob += ADDITIVE_VALUE
                    prob_cell_mapping[crew_move] = adj_cell.crew_probs.crew_prob/self.crew_search_data.normalize_probs
                    adj_cell.crew_probs.crew_prob = 0

                adj_cell.crew_probs.crew_prob += prob_cell_mapping[cell.cord]/total_moves
                if (beep_recv and adj_cell != self.curr_pos):
                    distance = get_manhattan_distance(self.curr_pos, crew_move)
                    adj_cell.crew_probs.crew_and_beep = adj_cell.crew_probs.crew_prob*LOOKUP_E[distance]
                    self.crew_search_data.beep_prob += adj_cell.crew_probs.crew_and_beep
                    updated_cells.add(crew_move)
                elif (not beep_recv and adj_cell != self.curr_pos):
                    distance = get_manhattan_distance(self.curr_pos, crew_move)
                    adj_cell.crew_probs.crew_and_no_beep = adj_cell.crew_probs.crew_prob*LOOKUP_NOT_E[distance]
                    self.crew_search_data.no_beep_prob += adj_cell.crew_probs.crew_and_no_beep
                    updated_cells.add(crew_move)
                else:
                    adj_cell.crew_probs.crew_prob = 0

        prob_cell_mapping.clear()
        self.crew_search_data.crew_cells = list(updated_cells)

    def update_crew_search_data(self):
        self.crew_search_data.normalize_probs = 0.0
        for cell_cord in self.crew_search_data.crew_cells:
            cell = self.ship.get_cell(cell_cord)
            if self.crew_search_data.is_beep_recv:
                cell.crew_probs.crew_prob = cell.crew_probs.crew_and_beep/self.crew_search_data.beep_prob
            else:
                cell.crew_probs.crew_prob = cell.crew_probs.crew_and_no_beep/self.crew_search_data.no_beep_prob

            self.crew_search_data.normalize_probs += cell.crew_probs.crew_prob
            if cell.zone_number not in self.zone_vs_zone_prob:
                self.zone_vs_zone_prob[cell.zone_number] = 0

            self.zone_vs_zone_prob[cell.zone_number] += cell.crew_probs.crew_prob

    def is_rescued(self):
        if self.curr_pos == self.curr_crew:
            self.logger.print(LOG_INFO, f"Bot has saved crew member at cell {self.curr_crew}")
            return True
        return False

    def remove_cell(self, rem_cell):
        crew_search_data = self.crew_search_data
        cell = self.ship.get_cell(rem_cell)
        crew_probs = cell.crew_probs
        crew_search_data.normalize_probs -= crew_probs.crew_prob
        cell.crew_probs.crew_prob = 0 # better to make this 0 and forget about this entirely
        self.crew_search_data.crew_cells.remove(rem_cell)
        self.logger.print(LOG_DEBUG, f"Removed {rem_cell} cell from possible crew cells {crew_search_data.crew_cells}")

    def is_continue_traversing(self):
        if not self.is_own_design or len(self.to_visit_list) == 0 or len(self.track_zones) == 0:
            return False

        dest = self.to_visit_list[0]
        zone_number = self.ship.get_cell(dest).zone_number
        if zone_number not in self.track_zones:
            return False

        if not zone_number in self.zone_vs_zone_prob:
            for element in self.track_zones[zone_number][0]:
                if element in self.to_visit_list:
                    self.to_visit_list.remove(element)

            del self.track_zones[zone_number]
            return False

        if self.zone_vs_zone_prob[zone_number] > self.track_zones[zone_number][1]:
            self.to_visit_list.clear()

        return True

    def calculate_best_path(self):
        if self.traverse_path:
            return True

        self.is_continue_traversing()

        if not self.to_visit_list:
            dummy, self.to_visit_list = self.shortest_route(self.get_best_zone())
            index = self.to_visit_list.index(self.curr_pos)
            self.to_visit_list.pop(index)
            if index > 0:
                self.to_visit_list.reverse()

        most_prob_cell = self.to_visit_list.pop(0)
        self.traverse_path = self.astar_search_path(most_prob_cell, [])
        if self.traverse_path:
            return True

        return False

    def start_rescue(self): # where it all begins....
        total_moves = total_iter = 0
        bot_moved = keep_moving = False
        init_distance = self.rescue_info()
        self.calc_initial_search_data()
        self.logger.print_all_crew_data(LOG_DEBUG, self)

        self.logger.print_heat_map(self.ship.grid, self.crew_search_data.is_beep_recv, self.curr_pos)
        # self.logger.print_heat_map(self.ship.grid, self.alien_evasion_data.is_beep_recv, self.curr_pos, False)
        while (True): # Keep trying till you find the crew
            if total_iter >= 1000:
                return init_distance, total_iter, total_moves, BOT_STUCK, 0

            total_iter += 1
            self.handle_alien_beep()
            self.handle_crew_beep()

            """
                Computation for aliens does not begin until the first alien beep is heard, this helps us to reduce the total compuations
            """
            if self.alien_evasion_data.beep_count > 0:
                if self.alien_evasion_data.beep_count == 1 and self.alien_evasion_data.is_beep_recv: # No need to init more than once...
                    self.alien_evasion_data.init_alien_calcs(self.curr_pos) # Initial alien cell will be the cells right outside & inside the alien zone, i.e., 2k+2 & 2k+1

                self.alien_evasion_data.beep_prob = 0 # reset probs
                self.alien_evasion_data.alien_movement_cells = set()
                self.compute_likely_alien_movements()
                self.alien_evasion_data.present_alien_cells = list(self.alien_evasion_data.alien_movement_cells)

            self.compute_likely_bot_movements()
            self.zone_vs_zone_prob.clear()
            if self.alien_evasion_data.beep_count > 0:
                self.update_alien_data()

            self.update_crew_search_data()
            self.logger.print_heat_map(self.ship.grid, self.crew_search_data.is_beep_recv, self.curr_pos)
            # self.logger.print_heat_map(self.ship.grid, self.alien_evasion_data.is_beep_recv, self.curr_pos, False)

            if self.calculate_best_path():
                if self.move_bot():
                    if self.is_rescued():
                        return init_distance, total_iter, total_moves, BOT_SUCCESS, 1
                    elif self.is_caught:
                        return init_distance, total_iter, total_moves, BOT_FAILED, 0

                    self.is_bot_moved = True
                    bot_moved = True
                    total_moves += 1
                    if self.curr_pos in self.crew_search_data.crew_cells:
                        self.remove_cell(self.curr_pos) # current cell can be ignored from all possible crew_cells
                else:
                    bot_moved = keep_moving = False

            if self.move_crew():
                return init_distance, total_iter, total_moves, BOT_SUCCESS, 1

            if self.ship.move_aliens(self):
                return init_distance, total_iter, total_moves, BOT_FAILED, 0


"""Simulation & Testing logic begins"""

BOT_NAMES = {
    0 : "bot_1",
    1 : "bot_2",
    2 : "bot_3",
    3 : "bot_4",
    4 : "bot_5",
    5 : "bot_6",
    6 : "bot_7",
    7 : "bot_8"
}

"""
    Responsible for updating the "alpha" / "k" for each worker pool
"""
def update_lookup(data, is_k):
    global LOOKUP_E, LOOKUP_NOT_E, ALPHA, ALIEN_ZONE_SIZE
    if is_k:
        ALIEN_ZONE_SIZE = data
    else:
        ALPHA = data

    LOOKUP_E = [(pow(exp, (-1*ALPHA*(i - 1)))) for i in range(GRID_SIZE*2 + 1)]
    LOOKUP_NOT_E = [(1-LOOKUP_E[i]) for i in range(GRID_SIZE*2 + 1)]

def bot_factory(itr, ship, log_level = LOG_NONE):
    if (itr == 0):
        return Bot_1(ship, log_level)
    elif (itr == 1):
        return Bot_2(ship, log_level)
    elif (itr == 2):
        return Bot_3(ship, log_level)
    elif (itr == 3):
        return Bot_4(ship, log_level)
    elif (itr == 4):
        return Bot_5(ship, log_level)
    elif (itr == 5):
        return Bot_6(ship, log_level)
    elif (itr == 6):
        return Bot_7(ship, log_level)
    elif (itr == 7):
        return Bot_8(ship, log_level)
    return ParentBot(ship, log_level)

def get_bonus(itr, ship, log_level = LOG_NONE):
    return Bonus_Bot(ship, log_level)

def test_bonus(log_level = LOG_INFO):
    update_lookup(ALPHA, False)
    ship = Ship(GRID_SIZE, log_level)
    ship.place_players()
    bot = get_bonus(0, ship, log_level)
    print(bot.start_rescue())
    del bot
    del ship

# Simple test function
def run_test(log_level = LOG_INFO):
    update_lookup(ALPHA, False)
    for itr in range(1):
        ship = Ship(GRID_SIZE, log_level)
        ship.place_players()
        for i in range(TOTAL_BOTS):
            print(BOT_NAMES[i], i)
            begin = time()
            bot = bot_factory(i, ship, log_level)
            print(bot.start_rescue())
            end = time()
            print(end - begin)
            del bot
            ship.reset_grid()
        del ship

class FINAL_OUT:
    def __init__(self) -> None:
        self.distance = 0
        self.total_iter = 0
        self.total_moves = 0
        self.idle_moves = 0
        self.success = 0
        self.success_steps = 0
        self.failure = 0
        self.failure_steps = 0
        self.stuck = 0
        self.stuck_steps = 0
        self.time_taken = 0.0
        self.crews_saved = 0
        pass

# Runs n number of iteration for each bot for given alpha value
def run_sim(args):
    iterations_range = args[0]
    data_range = args[1]
    is_bonus = args[2]
    data_dict = dict()
    itr_data = []
    is_k = True
    if "alpha" in data_range:
        itr_data = data_range["alpha"]
        is_k = False
    else:
        itr_data = data_range["k"]

    total_bots = TOTAL_BOTS
    bot_fac_func = bot_factory
    if is_bonus:
        total_bots = 1
        bot_fac_func = get_bonus

    # varying_data = "k" if is_k else "alpha"
    for data in itr_data:
        update_lookup(data, is_k)
        # print(f"Verifying update (alpha vs k) for variable {varying_data}::{ALPHA}::{ALIEN_ZONE_SIZE}")
        temp_data_set = [FINAL_OUT() for j in range(total_bots)]
        for itr in range(iterations_range):
            # print(itr+1, end = '\r') # MANNNYYY LINES PRINTED ON ILAB ;-;
            ship = Ship(GRID_SIZE)
            ship.place_players()
            for bot_no in range(total_bots):
                bot = bot_fac_func(bot_no, ship)
                begin = time()
                ret_vals = bot.start_rescue()
                end = time()
                temp_data_set[bot_no].distance += ret_vals[0]
                temp_data_set[bot_no].total_iter += ret_vals[1]
                temp_data_set[bot_no].total_moves += ret_vals[2]
                if ret_vals[3] == BOT_SUCCESS:
                    temp_data_set[bot_no].success += 1
                    temp_data_set[bot_no].success_steps += ret_vals[1]
                elif ret_vals[3] == BOT_FAILED:
                    temp_data_set[bot_no].failure += 1
                    temp_data_set[bot_no].failure_steps += ret_vals[1]
                else:
                    temp_data_set[bot_no].stuck += 1
                    temp_data_set[bot_no].stuck_steps += ret_vals[1]
                temp_data_set[bot_no].crews_saved += ret_vals[4]
                temp_data_set[bot_no].time_taken += (end-begin)
                ship.reset_grid()
                del bot
            del ship
        data_dict[data] = temp_data_set
    return data_dict

# Creates "n" process, and runs multiple simulation for same value of alpha simulataenously
def run_multi_sim(data_range, is_print = False, is_bonus = False):
    begin = time()
    result_dict = dict()
    total_bots = TOTAL_BOTS
    if is_bonus:
        total_bots = 1
    data_set = [FINAL_OUT() for j in range(total_bots)]
    processes = []
    print(f"Iterations begin...")
    core_count = cpu_count()
    total_iters = ceil(TOTAL_ITERATIONS/core_count)
    actual_iters = total_iters * core_count
    total_data = []
    for itr in range(core_count):
        total_data.append((total_iters, data_range, is_bonus))

    with Pool(processes=core_count) as p:
        result = p.map(run_sim, total_data)
        for temp_alpha_dict in result:
            for key, value in temp_alpha_dict.items():
                if key not in result_dict:
                    result_dict[key] = value
                else:
                    for i, val_range in enumerate(value):
                        result_dict[key][i].distance += value[i].distance
                        result_dict[key][i].total_iter += value[i].total_iter
                        result_dict[key][i].total_moves += value[i].total_moves
                        result_dict[key][i].success += value[i].success
                        result_dict[key][i].success_steps += value[i].success_steps
                        result_dict[key][i].failure += value[i].failure
                        result_dict[key][i].failure_steps += value[i].failure_steps
                        result_dict[key][i].stuck += value[i].stuck
                        result_dict[key][i].stuck_steps += value[i].stuck_steps
                        result_dict[key][i].crews_saved += value[i].crews_saved
                        result_dict[key][i].time_taken += value[i].time_taken

    for key, resc_val in result_dict.items():
        for itr, value in enumerate(resc_val):
            value.distance /= actual_iters
            value.total_iter /= actual_iters
            value.total_moves /= actual_iters
            value.idle_moves = value.total_iter - value.total_moves
            if value.success:
                value.success_steps /= value.success
            value.success /= actual_iters
            if value.failure:
                value.failure_steps /= value.failure
            value.failure /= actual_iters
            if value.stuck:
                value.stuck_steps /= value.stuck
            value.stuck /= actual_iters
            value.crews_saved /= actual_iters
            value.time_taken /= actual_iters
    end = time()

    is_const_alpha = True
    if "alpha" in data_range:
        is_const_alpha = False

    if (is_print):
        for key, resc_val in result_dict.items():
            if is_const_alpha:
                print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for k {key}, and alpha {ALPHA}")
            else:
                print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for alpha {key}, and k {ALIEN_ZONE_SIZE}")
            print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" % ("Bot", "Success Rate", "Failure Rate", "Stuck", "Distance", "Crews Saved", "Success steps", "Failure steps", "Stuck steps", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
            for itr, value in enumerate(resc_val):
                print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" % (BOT_NAMES[itr], value.success, value.failure, value.stuck, value.distance, value.crews_saved, value.success_steps, value.failure_steps, value.stuck_steps, value.total_iter, value.idle_moves, value.total_moves, value.time_taken))
    else:
        if is_const_alpha:
            print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for k {result_dict.keys()}, and alpha {ALPHA}")
        else:
            print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for alpha {result_dict.keys()}, and k {ALIEN_ZONE_SIZE}")

    del processes
    return result_dict

# Runs multiple simulations for multiple values of alpha concurrently
def compare_multiple_alpha(is_bonus = False):
    global ALPHA
    ALPHA = 0.1
    alpha_range = [round(ALPHA + (ALPHA_STEP_INCREASE * i), 2) for i in range(MAX_ALPHA_ITERATIONS)]
    alpha_dict = run_multi_sim({"alpha" : alpha_range}, False, is_bonus)
    print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" % ("Bot", "Success Rate", "Failure Rate", "Stuck", "Distance", "Crews Saved", "Success steps", "Failure steps", "stuck Steps", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
    for alpha, resc_val in alpha_dict.items():
        print(f"{'*'*82}{alpha}{'*'*82}")
        for itr, value in enumerate(resc_val):
            print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" %  (BOT_NAMES[itr], value.success, value.failure, value.stuck, value.distance, value.crews_saved, value.success_steps, value.failure_steps, value.stuck_steps, value.total_iter, value.idle_moves, value.total_moves, value.time_taken))

def compare_multiple_k(is_bonus = False):
    global ALIEN_ZONE_SIZE
    ALIEN_ZONE_SIZE = 2
    k_range = [round(ALIEN_ZONE_INCREASE + (ALIEN_ZONE_INCREASE * i), 2) for i in range(MAX_K_ITERATIONS)]
    k_dict = run_multi_sim({"k" : k_range}, False, is_bonus)
    print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" % ("Bot", "Success Rate", "Failure Rate", "Stuck", "Distance", "Crews Saved", "Success steps", "Failure steps", "stuck Steps", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
    for k, resc_val in k_dict.items():
        print(f"{'*'*82}{k}{'*'*82}")
        for itr, value in enumerate(resc_val):
            print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" %  (BOT_NAMES[itr], value.success, value.failure, value.stuck, value.distance, value.crews_saved, value.success_steps, value.failure_steps, value.stuck_steps, value.total_iter, value.idle_moves, value.total_moves, value.time_taken))

def run_multi_iters(is_bonus = False):
    compare_multiple_alpha(is_bonus)
    compare_multiple_k(is_bonus)

if __name__ == '__main__':
    # test_bonus()
    # run_test()
    # run_multi_sim({"alpha" : [ALPHA]}, True)
    # run_multi_sim({"k" : [ALIEN_ZONE_SIZE]}, True)
    run_multi_iters()
    run_multi_iters(True)
