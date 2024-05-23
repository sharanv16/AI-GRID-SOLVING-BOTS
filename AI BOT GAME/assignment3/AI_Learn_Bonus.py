import AI_Proj3
import AI_Bonus3
import run_simulations
from multiprocessing import cpu_count
from run_simulations import DETAILS
from pandas import read_csv
from ast import literal_eval
import torch
from matplotlib import pylab as plt
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from time import time
import numpy as np
from os import path
import random
import csv

MAX_TRAIN = 1000
MAX_TEST = 1000
MAX_PROCESS = int(cpu_count()/2)
GENERAL_TRAIN = int(96/MAX_PROCESS)
GENERAL_TEST = int(108/MAX_PROCESS)

AI_Proj3.GRID_SIZE = 7
AI_Proj3.TOTAL_ELEMENTS = 5
AI_Proj3.RAND_CLOSED_CELLS = 3
AI_Proj3.CONVERGENCE_LIMIT = 1
FULL_GRID_STATE = AI_Proj3.TOTAL_ELEMENTS*AI_Proj3.GRID_SIZE**2
LEARNING_RATE=1e-3

# H_LAYERS = [int(FULL_GRID_STATE*2), int(FULL_GRID_STATE*2)]
H_LAYERS = [int(FULL_GRID_STATE*2), int(FULL_GRID_STATE/2), int(FULL_GRID_STATE/3), int(FULL_GRID_STATE/3), int(FULL_GRID_STATE/2), int(FULL_GRID_STATE*2)]
BOT_ACTIONS = 9
AI_Proj3.VISUALIZE = True

IS_DEBUG = False
run_simulations.IS_BONUS = True
run_simulations.TOTAL_CONFIGS = 1

ACTIONS_ID = {
"IDLE" : int(0),
"NORTH" : int(1),
"SOUTH" : int(5),
"EAST" : int(3),
"WEST" : int(7),
"NORTH_EAST" : int(2),
"NORTH_WEST" : int(8),
"SOUTH_EAST" : int(4),
"SOUTH_WEST" : int(6)
}

class HIDDEN_UNITS(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(HIDDEN_UNITS, self).__init__()
        self.activation = activation
        self.nn = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.nn(x)
        out = self.activation(out)
        return out

class QModel(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, activation = F.leaky_relu):
        super(QModel, self).__init__()
        assert type(hidden_layers) is list
        self.hidden_units = nn.ModuleList()
        self.in_channels = in_channels
        prev_layer = in_channels
        for hidden in hidden_layers:
            self.hidden_units.append(HIDDEN_UNITS(prev_layer, hidden, activation))
            prev_layer = hidden
        self.final_unit = nn.Linear(prev_layer, out_channels)

    def forward(self, x):
        out = x.view(-1,self.in_channels).float()
        for unit in self.hidden_units:
            out = unit(out)
        out = self.final_unit(out)
        return out

class LEARN_CONFIG(AI_Bonus3.ALIEN_SHIP):
    def __init__(self, model_data, is_import = False):
        super(LEARN_CONFIG, self).__init__(is_import)
        self.q_model = model_data
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.losses = []
        self.total_failure_moves = self.total_success_moves = 0

    def import_ship(self, file_name="layout.csv"):
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for data in reader:
                closed_cells_str = literal_eval(str(data))
                for cell_str in closed_cells_str:
                    cell = literal_eval(cell_str)
                    self.closed_cells.append(cell)
                    self.open_cells.remove(cell)
                    self.set_state(cell, AI_Proj3.CLOSED_CELL)

        self.set_teleport()
        self.place_players()

    def print_losses(self):
        # plt.figure(figsize=(10,7))
        # plt.plot(self.losses)
        # plt.xlabel("Epochs",fontsize=22)
        # plt.ylabel("Loss",fontsize=22)
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        print(self.total_failure_moves, self.total_success_moves, self.total_failure_moves/(self.total_failure_moves+self.total_success_moves), len(self.losses))
        self.total_failure_moves = self.total_success_moves = 0
        self.losses.clear()

    def get_action(self, bot_cell, bot_move):
        val = -1
        if bot_cell and bot_move:
            delta_x = bot_move[0] - bot_cell[0]
            delta_y = bot_move[1] - bot_cell[1]

            if delta_x == 0 and delta_y == 0:
                val = ACTIONS_ID["IDLE"]
            elif delta_x == 0:
                val = ACTIONS_ID["NORTH"] if delta_y > 0 else ACTIONS_ID["SOUTH"]
            elif delta_y == 0:
                val = ACTIONS_ID["EAST"] if delta_x > 0 else ACTIONS_ID["WEST"]
            elif delta_x > 0:
                val = ACTIONS_ID["NORTH_EAST"] if delta_y > 0 else ACTIONS_ID["SOUTH_EAST"]
            else:
                val = ACTIONS_ID["NORTH_WEST"] if delta_y > 0 else ACTIONS_ID["SOUTH_WEST"]
        return val

    def get_ship_layout(self, bot_pos, crew_pos, alien_pos):
        rows = []
        for i in range(self.size):
            cols = []
            for j in range(self.size):
                curr_state = self.get_state((i, j))
                states = []
                states.append(1 if (i, j) == bot_pos else 0)
                states.append(1 if (i, j) == crew_pos else 0)
                states.append(-1 if (i, j) == alien_pos else 0)
                states.append(1 if curr_state == AI_Proj3.TELEPORT_CELL else 0)
                states.append(-1 if curr_state == AI_Proj3.CLOSED_CELL else 0)
                cols.extend(states)

            rows.extend(cols)

        final = (np.asarray(rows, dtype=np.float64))
        return final

class Q_BOT(AI_Bonus3.ALIEN_CONFIG):
    def __init__(self, ship, epsilon):
        super(Q_BOT, self).__init__(ship)
        self.old_bot_pos = ()
        self.old_crew_pos = ()
        self.state_1 = self.ship.get_ship_layout(self.local_bot_pos, self.local_crew_pos, self.local_alien_pos)
        self.tensor_1 = torch.from_numpy(self.state_1).float()
        self.state_2 = np.array([])
        self.is_train = True
        self.epsilon = epsilon
        self.action_list = []
        self.local_check = 0

    def make_action(self):
        next_move = self.local_all_moves[self.action_no]
        next_pos = (self.local_bot_pos[0] + next_move[0], self.local_bot_pos[1] + next_move[1])
        if 0 < next_pos[0] < self.ship.size and 0 < next_pos[1] < self.ship.size:
            state = self.ship.get_state(next_pos)
            if (self.is_train and state != AI_Proj3.CLOSED_CELL and state != AI_Proj3.CREW_CELL and state != AI_Bonus3.ALIEN_CELL) or (not self.is_train and state != AI_Proj3.CLOSED_CELL and state != AI_Proj3.CREW_CELL and state != AI_Bonus3.ALIEN_CELL):
                self.make_bot_move(next_pos)

    def calc_loss(self):
        bot_pos = self.local_bot_pos
        crew_pos = self.local_crew_pos
        alien_pos = self.local_alien_pos
        self.state_2 = self.ship.get_ship_layout(bot_pos, crew_pos, alien_pos)
        self.tensor_2 = torch.from_numpy(self.state_2).float()
        with torch.no_grad():
            possibleQs = self.ship.q_model(self.tensor_2)

        if self.action_no != self.best_action:
            self.ship.total_failure_moves += 1
        else:
            self.ship.total_success_moves += 1

        output = torch.Tensor([self.best_action]).long()
        loss = self.ship.loss_fn(self.q_vals, output)

        if IS_DEBUG and (self.total_moves % 200 == 0 or self.local_check):
            self.local_check += 1
            self.ship.print_ship()
            print(self.best_action, self.local_all_moves[self.best_action], self.action_no, self.local_all_moves[self.action_no], self.best_move)
            print(self.q_vals, output, loss)
            if self.local_check == 5:
                self.local_check = 0

        if (self.is_train):
            self.ship.optimizer.zero_grad()
            loss.backward()
            self.ship.losses.append(loss.item())
            self.ship.optimizer.step()
        else:
            loss.backward()
            self.ship.losses.append(loss.item())

        self.state_1 = self.state_2
        self.tensor_1 = self.tensor_2

    def process_q_learn(self):
        self.q_vals = self.ship.q_model(self.tensor_1)
        self.best_move = self.ship.best_policy_lookup[self.local_bot_pos][self.local_alien_pos][self.local_crew_pos]
        self.best_action = self.ship.get_action(self.local_bot_pos, self.best_move)
        if random.uniform(0, 1) < self.epsilon:
            self.action_no = self.best_action
            # self.action_no = np.random.randint(0,9)
        else:
            self.action_no = int(torch.argmax(self.q_vals).item())

        self.make_action()
        return self.move_alien() or self.move_crew()

    def move_bot(self):
        self.q_vals = self.ship.q_model(self.tensor_1)
        self.action_no = int(torch.argmax(self.q_vals).item())
        self.best_move = self.ship.best_policy_lookup[self.local_bot_pos][self.local_alien_pos][self.local_crew_pos]
        self.best_action = self.ship.get_action(self.local_bot_pos, self.best_move)
        self.make_action()

    def move_alien(self):
        ret_val = super(Q_BOT, self).move_alien()
        if ret_val:
            self.calc_loss()
        return ret_val

    def move_crew(self):
        ret_val = super(Q_BOT, self).move_crew()
        self.calc_loss()
        return ret_val

    def train_rescue(self):
        self.is_train = True
        self.total_moves = 0
        if self.ship.get_state(self.local_crew_pos) & AI_Proj3.TELEPORT_CELL:
            return self.total_moves, AI_Proj3.SUCCESS

        while(True):
            self.total_moves += 1
            if self.process_q_learn():
                return self.total_moves, AI_Proj3.SUCCESS

            if self.total_moves > 1000:
                return self.total_moves, AI_Proj3.FAILURE

    def test_rescue(self):
        self.is_train = False
        return self.start_rescue()

def t_bot(ship, is_train = True):
    epochs = MAX_TRAIN if is_train else MAX_TEST
    avg_moves = DETAILS()
    epsilon = 1.0
    update_epsilon = (1/epochs)
    for iters in range(epochs):
        # print(iters, end='\r')
        q_bot = Q_BOT(ship, epsilon)
        if is_train:
            moves, result = q_bot.train_rescue()
        else:
            moves, result = q_bot.test_rescue()

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
        del q_bot

        if epsilon > 0.1:
            epsilon -= update_epsilon

        ship.reset_positions()

    # print()
    return avg_moves

def single_sim(ship):
    final_data = run_simulations.run_sim([range(0, MAX_TEST), ship])

    run_simulations.print_header(MAX_TEST)
    for itr in range(run_simulations.TOTAL_CONFIGS):
        run_simulations.print_data(final_data[itr], itr, MAX_TEST)

def single_run():
    q_model = QModel(FULL_GRID_STATE, H_LAYERS, BOT_ACTIONS, F.relu)
    ship = LEARN_CONFIG(q_model)
    ship.perform_initial_calcs()
    run_simulations.print_data(t_bot(ship), 4, MAX_TRAIN)
    ship.print_losses()
    run_simulations.print_data(t_bot(ship, False), 4, MAX_TEST)
    ship.print_losses()
    single_sim(ship)
    del ship

def train(q_model, result_queue):
    avg_moves = DETAILS()
    for i in range(GENERAL_TRAIN):
        ship = LEARN_CONFIG(q_model)
        ship.perform_initial_calcs()
        avg_moves.update(t_bot(ship))
        ship.print_losses()
    avg_moves.get_avg(GENERAL_TRAIN)
    result_queue.put(avg_moves)

def test(q_model, result_queue):
    avg_moves = DETAILS()
    for i in range(GENERAL_TEST):
        ship = LEARN_CONFIG(q_model)
        ship.perform_initial_calcs()
        avg_moves.update(t_bot(ship, False))
        ship.print_losses()
    avg_moves.get_avg(GENERAL_TRAIN)
    result_queue.put(avg_moves)

def multi_run():
    processes = []
    q_model = QModel(FULL_GRID_STATE, H_LAYERS, BOT_ACTIONS)
    q_model.share_memory()
    print("Training data...")
    detail = DETAILS()
    result_queue = torch.multiprocessing.Queue()
    for rank in range(MAX_PROCESS):
        p = torch.multiprocessing.Process(target=train, args=(q_model, result_queue, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        detail.update(result_queue.get())
    run_simulations.print_data(detail, 4, MAX_PROCESS)

    print("Testing data...")
    detail = DETAILS()
    processes.clear()
    for rank in range(MAX_PROCESS):
        p = torch.multiprocessing.Process(target=test, args=(q_model, result_queue, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        detail.update(result_queue.get())
    run_simulations.print_data(detail, 4, MAX_PROCESS)

if __name__ == '__main__':
    begin = time()
    single_run()
    # multi_run()
    end = time()
    print(end-begin)
