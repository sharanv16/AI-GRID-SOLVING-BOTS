import AI_Proj3
from random import choice, random
from copy import deepcopy
from time import time

TOTAL_ELEMENTS = 5
ALIEN_CELL = 32
CAUGHT = 2

class ALIEN_SHIP(AI_Proj3.SHIP):
    def __init__(self, is_import = False):
        self.alien_pos = (0, 0)
        super(ALIEN_SHIP, self).__init__(is_import)
        self.max_time_steps = self.size**6*9*4*4
        self.global_min_max = -self.size**8*9*4*4

    def place_players(self):
        super(ALIEN_SHIP, self).place_players()
        while(True):
            self.alien_pos = choice(self.open_cells)
            if self.alien_pos == self.teleport_cell or self.search_path(self.alien_pos):
                break

        alien_state = (AI_Proj3.TELEPORT_CELL | ALIEN_CELL) if (self.alien_pos == self.teleport_cell) else ALIEN_CELL
        self.set_state(self.alien_pos, alien_state)
        self.open_cells.remove(self.alien_pos)

    def unset_global_states(self):
        del self.indi_states_lookup
        del self.time_lookup
        del self.alien_moves
        del self.crew_moves
        del self.bot_moves
        del self.c_manhattan_lookup
        del self.a_manhattan_lookup
        del self.all_possible_states

    def set_calcs_lookup(self):
        self.all_possible_states = dict()
        self.a_manhattan_lookup = dict()
        self.c_manhattan_lookup = dict()
        self.bot_moves = dict()
        self.crew_moves = dict()
        self.alien_moves = dict()
        for bot_r in range(self.size):
            for bot_c in range(self.size):
                bot_pos = (bot_r, bot_c)
                if self.get_state(bot_pos) == AI_Proj3.CLOSED_CELL:
                    continue

                self.all_possible_states[bot_pos] = dict()
                self.a_manhattan_lookup[bot_pos] = dict()
                self.c_manhattan_lookup[bot_pos] = dict()
                bot_movements = self.get_all_moves(bot_pos, ~AI_Proj3.CLOSED_CELL, False)
                bot_movements.append(bot_pos)
                self.bot_moves[bot_pos] = bot_movements
                alien_states = self.all_possible_states[bot_pos]
                a_distances = self.a_manhattan_lookup[bot_pos]
                c_distances = self.c_manhattan_lookup[bot_pos]
                for alien_r in range(self.size):
                    for alien_c in range(self.size):
                        alien_pos = (alien_r, alien_c)
                        if self.get_state(alien_pos) == AI_Proj3.CLOSED_CELL or alien_pos == bot_pos:
                            continue

                        alien_states[alien_pos] = list()
                        crew_states = alien_states[alien_pos]
                        a_distances[alien_pos] = AI_Proj3.get_manhattan_distance(alien_pos, bot_pos)
                        if alien_pos not in self.alien_moves:
                            self.alien_moves[alien_pos] = self.get_all_moves(alien_pos, ~AI_Proj3.CLOSED_CELL)

                        for crew_r in range(self.size):
                            for crew_c in range(self.size):
                                crew_pos = (crew_r, crew_c)
                                if self.get_state(crew_pos) == AI_Proj3.CLOSED_CELL or crew_pos == bot_pos:
                                    continue

                                if crew_pos == alien_pos:
                                    self.indi_states_lookup[bot_pos[0]][bot_pos[1]][alien_pos[0]][alien_pos[1]][crew_pos[0]][crew_pos[1]] = self.global_min_max
                                    self.time_lookup[bot_pos[0]][bot_pos[1]][alien_pos[0]][alien_pos[1]][crew_pos[0]][crew_pos[1]] = self.max_time_steps

                                crew_states.append(crew_pos)
                                c_distances[crew_pos] = AI_Proj3.get_manhattan_distance(crew_pos, bot_pos)
                                if crew_pos not in self.crew_moves:
                                    self.crew_moves[crew_pos] = self.get_all_moves(crew_pos, ~AI_Proj3.CLOSED_CELL)

    def set_state_details(self, total_level):
        super(ALIEN_SHIP, self).set_state_details(4)

    def get_final_player_moves(self, bot_pos, player_pos, player_type):
        curr_bot_distance = self.a_manhattan_lookup[bot_pos][player_pos] if player_type == ALIEN_CELL else self.c_manhattan_lookup[bot_pos][player_pos]
        all_moves = self.alien_moves[player_pos] if player_type == ALIEN_CELL else self.crew_moves[player_pos]
        new_movements = []
        if curr_bot_distance == 1:
            escape_moves = []
            max_dist = 0
            for player_move in all_moves:
                if player_move == bot_pos:
                    continue

                new_dist =  self.a_manhattan_lookup[bot_pos][player_move] if player_type == ALIEN_CELL else self.c_manhattan_lookup[bot_pos][player_move]
                escape_moves.append((player_move, new_dist))
                if max_dist < new_dist:
                    max_dist = new_dist

            for escape in escape_moves:
                if escape[1] == max_dist:
                    new_movements.append(escape[0])
        else:
            new_movements = list(all_moves)

        if not len(new_movements):
            new_movements.append(player_pos)

        return new_movements

    def perform_moves_iteration(self):
        moves_possible = {}
        prev_difference = 0.0
        for iters in range(self.ideal_iters_limit):
            avg_difference = 0.0
            for bot_pos in self.all_possible_states:
                for alien_pos in self.all_possible_states[bot_pos]:
                    for crew_pos in self.all_possible_states[bot_pos][alien_pos]:
                        if crew_pos == self.teleport_cell or alien_pos == crew_pos:
                            continue

                        min_max = 1
                        for bot_action in self.bot_moves[bot_pos]:
                            if bot_action == alien_pos or bot_action == crew_pos:
                                continue

                            action_time_step = 0.0
                            alien_movements = self.get_final_player_moves(bot_action, alien_pos, ALIEN_CELL)
                            crew_movements = self.get_final_player_moves(bot_action, crew_pos, AI_Proj3.CREW_CELL)
                            total_moves = len(alien_movements)*len(crew_movements)
                            total_move_probs = 1/total_moves
                            for alien_move in alien_movements:
                                for crew_move in crew_movements:
                                    if crew_move == self.teleport_cell:
                                        continue

                                    action_time_step += self.time_lookup[bot_action[0]][bot_action[1]][alien_move[0]][alien_move[1]][crew_move[0]][crew_move[1]]*total_move_probs

                            if action_time_step > min_max:
                                min_max = action_time_step

                        old_max = self.time_lookup[bot_pos[0]][bot_pos[1]][alien_pos[0]][alien_pos[1]][crew_pos[0]][crew_pos[1]]
                        self.time_lookup[bot_pos[0]][bot_pos[1]][alien_pos[0]][alien_pos[1]][crew_pos[0]][crew_pos[1]] = min_max
                        avg_difference += min_max - old_max

            convergence = prev_difference - avg_difference
            if convergence >= 0 and convergence < AI_Proj3.CONVERGENCE_LIMIT:
                self.ideal_iters_limit = iters
                break

            prev_difference = avg_difference

    def perform_value_iteration(self):
        self.best_policy_lookup = {}
        max_iters = 0
        for iters in range(self.ideal_iters_limit):
            current_iters = 0
            for bot_pos in self.all_possible_states:
                for alien_pos in self.all_possible_states[bot_pos]:
                    for crew_pos in self.all_possible_states[bot_pos][alien_pos]:
                        if crew_pos == self.teleport_cell or alien_pos == crew_pos:
                            continue

                        min_max = self.global_min_max
                        action_val = ()
                        all_policies = {}
                        for bot_action in self.bot_moves[bot_pos]:
                            if bot_action == alien_pos or bot_action == crew_pos:
                                continue

                            alien_movements = self.get_final_player_moves(bot_action, alien_pos, ALIEN_CELL)
                            crew_movements = self.get_final_player_moves(bot_action, crew_pos, AI_Proj3.CREW_CELL)
                            total_moves = len(alien_movements)*len(crew_movements)
                            total_move_probs = 1/total_moves
                            action_value = -1
                            for alien_move in alien_movements:
                                for crew_move in crew_movements:
                                    if crew_move == self.teleport_cell:
                                        continue

                                    crew_reward = self.time_lookup[bot_action[0]][bot_action[1]][alien_move[0]][alien_move[1]][crew_move[0]][crew_move[1]]
                                    new_state_value = self.indi_states_lookup[bot_action[0]][bot_action[1]][alien_move[0]][alien_move[1]][crew_move[0]][crew_move[1]]
                                    action_value += (total_move_probs*(new_state_value - crew_reward))

                            if action_value > min_max:
                                min_max = action_value
                                action_val = bot_action

                            all_policies[bot_action] = action_value

                        old_max = self.indi_states_lookup[bot_pos[0]][bot_pos[1]][alien_pos[0]][alien_pos[1]][crew_pos[0]][crew_pos[1]]
                        self.indi_states_lookup[bot_pos[0]][bot_pos[1]][alien_pos[0]][alien_pos[1]][crew_pos[0]][crew_pos[1]] = min_max
                        if iters == 0:
                            max_iters += 1
                        elif old_max - min_max < AI_Proj3.CONVERGENCE_LIMIT or (iters + 1) == self.ideal_iters_limit:
                            if bot_pos not in self.best_policy_lookup:
                                self.best_policy_lookup[bot_pos] = dict()

                            alien_data = self.best_policy_lookup[bot_pos]
                            if alien_pos not in self.best_policy_lookup[bot_pos]:
                                alien_data[alien_pos] = dict()

                            bot_policy = alien_data[alien_pos]
                            bot_policy[crew_pos] = action_val
                            current_iters += 1

            if current_iters == max_iters:
                break

    def reset_grid(self):
        super(ALIEN_SHIP, self).reset_grid()
        alien_state = (AI_Proj3.TELEPORT_CELL | ALIEN_CELL) if (self.alien_pos == self.teleport_cell) else ALIEN_CELL
        self.set_state(self.alien_pos, alien_state)

    def reset_positions(self):
        self.open_cells.append(self.alien_pos)
        super(ALIEN_SHIP, self).reset_positions()
    
    def reset_static_pos(self, bot_pos):
        self.open_cells.append(self.alien_pos)
        super(ALIEN_SHIP, self).reset_static_pos(bot_pos)
        while(True):
            self.alien_pos = choice(self.open_cells)
            if self.alien_pos == self.teleport_cell or self.search_path(self.alien_pos):
                break

        alien_state = (AI_Proj3.TELEPORT_CELL | ALIEN_CELL) if (self.alien_pos == self.teleport_cell) else ALIEN_CELL
        self.set_state(self.alien_pos, alien_state)
        self.open_cells.remove(self.alien_pos)

class ALIEN_CONFIG(AI_Proj3.BOT_CONFIG):
    def __init__(self, ship):
        super(ALIEN_CONFIG, self).__init__(ship)
        self.local_alien_pos = self.ship.alien_pos

    def append_move(self):
        if self.csv_data is None:
            return

        self.csv_data.append([self.local_bot_pos, self.local_crew_pos, self.local_alien_pos])

    def move_bot(self):
        bot_movements = self.ship.get_all_moves(self.local_bot_pos, AI_Proj3.OPEN_CELL | AI_Proj3.TELEPORT_CELL, False)
        bot_movements.append(self.local_bot_pos)
        self.best_move = self.ship.best_policy_lookup[self.local_bot_pos][self.local_alien_pos][self.local_crew_pos]
        if not self.best_move:
            return

        self.make_bot_move(self.best_move)

    def move_alien(self):
        neighbors = self.ship.get_all_moves(self.local_alien_pos, AI_Proj3.OPEN_CELL | AI_Proj3.TELEPORT_CELL | AI_Proj3.CREW_CELL)
        next_cell = self.get_next_move(neighbors)
        if next_cell is None:
            return False

        curr_state = AI_Proj3.TELEPORT_CELL if (self.local_alien_pos == self.ship.teleport_cell) else AI_Proj3.OPEN_CELL
        self.ship.set_state(self.local_alien_pos, curr_state)
        self.local_alien_pos = next_cell
        next_cell_state = self.ship.get_state(next_cell)
        next_state = ALIEN_CELL
        is_caught = False
        if self.local_alien_pos == self.ship.teleport_cell:
            next_state |= AI_Proj3.TELEPORT_CELL
        elif self.local_crew_pos == self.local_alien_pos:
            next_state |= AI_Proj3.CREW_CELL
            self.ship.set_state(next_cell, next_state)
            self.return_state = CAUGHT
            is_caught = True

        self.ship.set_state(next_cell, next_state)
        return is_caught

    def move_crew(self):
        neighbors = self.ship.get_all_moves(self.local_crew_pos, AI_Proj3.OPEN_CELL | AI_Proj3.TELEPORT_CELL | ALIEN_CELL)
        next_cell = self.get_next_move(neighbors)
        if next_cell is None:
            return False

        self.ship.set_state(self.local_crew_pos, AI_Proj3.OPEN_CELL)
        self.local_crew_pos = next_cell
        next_cell_state = self.ship.get_state(next_cell)
        next_state = AI_Proj3.CREW_CELL
        is_fin = False
        if self.local_crew_pos == self.local_alien_pos:
            next_state |= ALIEN_CELL
            self.return_state = CAUGHT
            is_fin = True
        elif self.local_crew_pos == self.ship.teleport_cell:
            next_state |= AI_Proj3.TELEPORT_CELL
            self.return_state = AI_Proj3.SUCCESS
            is_fin = True

        self.ship.set_state(next_cell, next_state)
        return is_fin
