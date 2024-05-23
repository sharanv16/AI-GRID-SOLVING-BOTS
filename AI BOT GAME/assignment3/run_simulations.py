import AI_Proj3
import AI_Bonus3
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from time import time
from math import ceil
import csv
import os
import shutil

TOTAL_ITERATIONS = 1000 # iterations for same ship layout and different bot/crew positions
IS_BONUS = True
TOTAL_CONFIGS = 1 if IS_BONUS else 2
MAX_CORES = cpu_count()

AI_Proj3.GRID_SIZE = 11
AI_Proj3.VISUALIZE = False
AI_Proj3.NO_CLOSED_CELLS = False
AI_Proj3.RAND_CLOSED_CELLS = 10
AI_Proj3.CONVERGENCE_LIMIT = 1 if IS_BONUS else 1e-4 # Small value to reduce time complexity

GENERALIZED_FOLDER="general_bonus" if IS_BONUS else "general"
SINGLE_FOLDER="single_bonus" if IS_BONUS else "single"
GENERALIZED_DATA="general.csv"
SINGLE_DATA="single.csv"
LAYOUT_DATA="layout.csv"
DATA_COLS=["Bot_Pos", "Crew_Pos", "Alien_Pos"] if IS_BONUS else ["Bot_Pos", "Crew_Pos"]
LAYOUT_COLS=["Closed_Cells", "Wall_Cells"]
SINGLE_MOVES=1000
GENERALIZED_SHIPS=400

BOT_NAME_LOOKUP = {
    0: "NO BOT",
    1: "BOT",
    2: "BOT LEARNT",
    3: "ALIEN",
    4: "ALIEN LEARNT"
}

class DETAILS:
    def __init__(self):
        self.success = self.failure = self.caught = 0.0
        self.s_moves = self.f_moves = self.c_moves = 0.0
        self.max_success = self.min_success = 0
        self.distance = 0.0
        self.dest_dist = 0.0
        self.best_cell = (-1, -1)

    def update_min_max(self, moves):
        if self.max_success < moves:
            self.max_success = moves

        if self.min_success > moves:
            self.min_success = moves

    def update(self, new_detail):
        self.s_moves += new_detail.s_moves
        self.success += new_detail.success
        self.f_moves += new_detail.f_moves
        self.failure += new_detail.failure
        self.caught += new_detail.caught
        self.c_moves += new_detail.c_moves
        self.distance += new_detail.distance
        self.dest_dist += new_detail.dest_dist
        self.update_min_max(new_detail.max_success)
        self.update_min_max(new_detail.min_success)

    def get_avg(self, total_itr):
        if self.success:
            self.s_moves /= self.success

        if self.failure:
            self.f_moves /= self.failure

        if self.caught:
            self.c_moves /= self.caught

        self.success /= total_itr
        self.failure /= total_itr
        self.caught /= total_itr
        self.distance /= total_itr
        self.dest_dist /= total_itr

def bot_fac(itr, myship):
    if IS_BONUS:
        return AI_Bonus3.ALIEN_CONFIG(myship)

    if itr % TOTAL_CONFIGS  == 0:
        return AI_Proj3.NO_BOT_CONFIG(myship)
    else:
        return AI_Proj3.BOT_CONFIG(myship)

def run_sim(args):
    if len(args) == 1:
        ship = AI_Bonus3.ALIEN_SHIP() if IS_BONUS else AI_Proj3.SHIP()
        ship.perform_initial_calcs()
    else:
        ship = args[1]

    avg_moves = [DETAILS() for itr in range(TOTAL_CONFIGS)]
    for _ in args[0]:
        # print(_, end = "\r")
        dest_dist = AI_Proj3.get_manhattan_distance(ship.crew_pos, ship.teleport_cell)
        for itr in range(TOTAL_CONFIGS):
            test_bot = bot_fac(itr, ship)
            moves, result = test_bot.start_rescue()
            ship.reset_grid()
            if result == 1:
                avg_moves[itr].update_min_max(moves)
                avg_moves[itr].s_moves += moves
                avg_moves[itr].success += 1
            elif result == 2:
                avg_moves[itr].c_moves += moves
                avg_moves[itr].caught += 1
            else:
                avg_moves[itr].f_moves += moves
                avg_moves[itr].failure += 1

            distance = 0 if test_bot.__class__ is AI_Proj3.NO_BOT_CONFIG else AI_Proj3.get_manhattan_distance(ship.bot_pos, ship.crew_pos)
            avg_moves[itr].distance += distance
            avg_moves[itr].dest_dist += dest_dist
            del test_bot

        ship.reset_positions()

    # print()
    del ship
    return avg_moves

def print_header(total_itr = TOTAL_ITERATIONS):
    print("Total iterations performed for layout is", total_itr)
    print("%13s %18s %18s %18s %18s %18s %18s %18s %18s %18s %18s" % ("Name", "Avg Suc Moves", "Success Rate", "Min Suc. Moves", "Max Suc. Moves", "Avg Caught Moves", "Caught Rate", "Avg Fail Moves", "Failure Rate", "Avg Bot Crew Dist", "Crew Teleport Dist"))

def print_data(final_data, itr, total_itr = TOTAL_ITERATIONS):
    if IS_BONUS and itr == 0:
        itr = 3

    final_data.get_avg(total_itr)
    print(("%13s %18s %18s %18s %18s %18s %18s %18s %18s %18s %18s" % (BOT_NAME_LOOKUP[itr], final_data.s_moves, final_data.success, final_data.min_success, final_data.max_success, final_data.c_moves, final_data.caught, final_data.f_moves, final_data.failure, final_data.distance, final_data.dest_dist)))

def run_multi_sim():
    core_count = MAX_CORES
    arg_data = [[range(0, TOTAL_ITERATIONS)] for i in range(core_count)]
    avg_moves = [[DETAILS() for itr in range(TOTAL_CONFIGS)] for _ in range(core_count)]
    with Pool(processes=core_count) as p:
        for layout, final_data in enumerate(p.map(run_sim, arg_data)):
            curr_ship = avg_moves[layout]
            for bot_no, data in enumerate(final_data):
                curr_ship[bot_no].update(data)

        print_header()
        for layout in range(core_count):
            print("Layout no. :: ", layout)
            curr_ship = avg_moves[layout]
            for itr in range(TOTAL_CONFIGS):
                print_data(curr_ship[itr], itr)

def single_sim(total_itr):
    final_data = run_sim([range(0, total_itr)])

    print_header(total_itr)
    for itr in range(TOTAL_CONFIGS):
        print_data(final_data[itr], itr, total_itr)

def single_run():
    ship = AI_Bonus3.ALIEN_SHIP() if IS_BONUS else AI_Proj3.SHIP()
    ship.perform_initial_calcs()
    ship.print_ship()
    for itr in range(TOTAL_CONFIGS):
        test_bot = bot_fac(itr, ship)
        print(test_bot.start_rescue())
        ship.reset_grid()

def create_file(file_name, headings):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headings)

    return file_name

def generate_data(args_list):
    for args in args_list:
        file_name = args[0]
        layout_file = args[1]
        ship = AI_Bonus3.ALIEN_SHIP() if IS_BONUS else AI_Proj3.SHIP()
        ship.perform_initial_calcs()
        write_ship_layout(ship, layout_file)
        for _ in range(ceil(SINGLE_MOVES/MAX_CORES)):
            bot_config = AI_Bonus3.ALIEN_CONFIG(ship) if IS_BONUS else AI_Proj3.BOT_CONFIG(ship)
            bot_config.start_data_collection(file_name)
            del bot_config
            ship.reset_positions()
        del ship

def get_generalized_data():
    print("get_generalized_data...")
    create_folder(GENERALIZED_FOLDER)
    arg_data = []
    ship_per_thread = ceil(GENERALIZED_SHIPS/MAX_CORES)
    count = 0
    for t in range(MAX_CORES):
        per_thread_data = []
        for i in range(ship_per_thread):
            inner_data = []
            general_file = os.path.join(GENERALIZED_FOLDER, str(count) + "_" + GENERALIZED_DATA)
            general_layout = os.path.join(GENERALIZED_FOLDER, str(count) + "_" + LAYOUT_DATA)
            inner_data.append(general_file)
            inner_data.append(general_layout)
            create_file(general_file, DATA_COLS)
            create_file(general_layout, LAYOUT_COLS)
            per_thread_data.append(inner_data)
            count += 1
        arg_data.append(per_thread_data)

    with Pool(processes=MAX_CORES) as p:
        p.map(generate_data, arg_data)

def generate_same_data(args):
    file_name = args[0]
    ship = deepcopy(args[1])
    for _ in range(ceil(SINGLE_MOVES/MAX_CORES)):
        bot_config = AI_Bonus3.ALIEN_CONFIG(ship) if IS_BONUS else AI_Proj3.BOT_CONFIG(ship)
        bot_config.start_data_collection(file_name)
        del bot_config
        ship.reset_positions()

    del ship

def write_ship_layout(ship, file_name):
    create_file(file_name, LAYOUT_COLS)
    total_closed = []
    closed = []
    for cell in ship.closed_cells:
        closed.append(cell)
    total_closed.append(closed)
    closed = []
    for cell in ship.wall_cells:
        closed.append(cell)
    total_closed.append(closed)
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(total_closed)

def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)

def get_single_data():
    print("get_single_data...")
    create_folder(SINGLE_FOLDER)
    ship = AI_Bonus3.ALIEN_SHIP() if IS_BONUS else AI_Proj3.SHIP()
    ship.perform_initial_calcs()
    arg_data = [["output_"+str(i)+".csv", ship] for i in range(MAX_CORES)]
    single_file = os.path.join(SINGLE_FOLDER, SINGLE_DATA)
    create_file(single_file, DATA_COLS)
    layout_file = os.path.join(SINGLE_FOLDER, LAYOUT_DATA)
    write_ship_layout(ship, layout_file)

    with open(single_file, 'a', newline='') as csvfile:
        with Pool(processes=MAX_CORES) as p:
            p.map(generate_same_data, arg_data)
            for args in arg_data:
                file_name = args[0]
                with open(file_name, mode ='r') as read_file:
                    csv_file = csv.reader(read_file)
                    for lines in csv_file:
                        writer = csv.writer(csvfile)
                        writer.writerow(lines)
                os.remove(file_name)

    del ship

def test_multiple_bot_pos():
    print_header()
    ship = AI_Bonus3.ALIEN_SHIP() if IS_BONUS else AI_Proj3.SHIP()
    ship.perform_initial_calcs()
    all_open = list(ship.open_cells)
    all_open.append(ship.bot_pos)
    all_open.append(ship.crew_pos)
    best_details = DETAILS()
    best_details.s_moves = 500
    for cell in all_open:
        if not (cell == ship.teleport_cell or ship.search_path(cell)):
            continue

        ship.reset_static_pos(cell)
        avg_moves = DETAILS()
        for itr in range(TOTAL_ITERATIONS):
            bot = AI_Bonus3.ALIEN_CONFIG(ship) if IS_BONUS else AI_Proj3.BOT_CONFIG(ship)
            moves, result = bot.start_rescue()
            if result == 1:
                avg_moves.update_min_max(moves)
                avg_moves.s_moves += moves
                avg_moves.success += 1
            elif result == 2:
                avg_moves.c_moves += moves
                avg_moves.caught += 1
            else:
                avg_moves.f_moves += moves
                avg_moves.failure += 1

            avg_moves.distance += AI_Proj3.get_manhattan_distance(ship.bot_pos, ship.crew_pos)
            avg_moves.dest_dist += AI_Proj3.get_manhattan_distance(ship.crew_pos, ship.teleport_cell)
            ship.reset_static_pos(cell)
        print(f"Data for crew pos::{cell}")
        print_data(avg_moves, 1)
        if best_details.s_moves > avg_moves.s_moves:
            best_details = avg_moves
            best_details.best_cell = cell

    print(f"Most optimal crew cell for current ship::{best_details.best_cell}")
    print_data(best_details, 1, 1)
    ship.reset_static_pos(best_details.best_cell)
    print("Ship Layout")
    ship.print_ship()


if __name__ == '__main__':
    begin = time()
    # test_multiple_bot_pos()
    # single_run()
    # single_sim(TOTAL_ITERATIONS)
    # run_multi_sim()
    get_single_data()
    # get_generalized_data()
    end = time()
    print(end-begin)
