import pandas as pd
import numpy as np
import math
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from ast import literal_eval
import AI_Proj3
import run_simulations


import warnings
warnings.filterwarnings('ignore')

is_general = True

label_encoder = LabelEncoder()
closed_cells_encoder = LabelEncoder()
walls_encoder = LabelEncoder()

def parse_tuple(cell):
    parts = cell.strip('()').split(',')
    return tuple(int(part.strip()) for part in parts)

def clean_Bot_Move(row, df):
    crew_cell = parse_tuple(row['Crew_Pos'])
    if crew_cell[0] == 5 and crew_cell[1] == 5:
        return None
    else:
        next_row_index = row.name + 1  # Get the index of the next row
        if next_row_index < len(df):
            return df.at[next_row_index, 'Bot_Pos']  # Return Bot_Cell value from the next row
            
def parse_coordinates(coord_str):
    if coord_str:
        x, y = map(int, coord_str.strip("()").split(","))
        return x, y
    else:
        return None

def calculate_direction(row):
    bot_cell = parse_coordinates(row['Bot_Cell'])
    bot_move = parse_coordinates(row['Bot_Move'])

    if bot_cell and bot_move:
        delta_x = bot_move[0] - bot_cell[0]
        delta_y = bot_move[1] - bot_cell[1]

        if delta_x == 0 and delta_y == 0:
            return "No movement"
        elif delta_x == 0:
            return "North" if delta_y > 0 else "South"
        elif delta_y == 0:
            return "East" if delta_x > 0 else "West"
        elif delta_x > 0:
            return "Northeast" if delta_y > 0 else "Southeast"
        else:
            return "Northwest" if delta_y > 0 else "Southwest"
    else:
        return "Invalid coordinates"

def map_coordinates_to_integer(row,celltype):
    cell = parse_coordinates(row[celltype])
    cols = 11
    return cell[0] * cols + cell[1] + 1

def encode_list_of_tuples(lst):
    return ','.join([str(x) for x in lst])

def convert_tuples(cell):
    return literal_eval(cell)

def lengthSquare(X, Y):
    xDiff = X[0] - Y[0]
    yDiff = X[1] - Y[1]
    return xDiff * xDiff + yDiff * yDiff

def getAngle(a, b):
    c = (5, 5)
    a2 = lengthSquare(a, c)
    b2 = lengthSquare(b, c)
    c2 = lengthSquare(a, b)
    return math.acos((a2 + b2 - c2) / (2 * math.sqrt(a2) * math.sqrt(b2)))#(math.acos((a2 + b2 - c2) / (2 * math.sqrt(a2) * math.sqrt(b2))) * 180 / math.pi);

def parse_angles(row):
    crew_cell = parse_tuple(row['Crew_Cell'])
    bot_cell = parse_tuple(row['Bot_Cell'])
    return getAngle(bot_cell, crew_cell)

def process_data():
    path = "alien_final_data.csv"  
    df = pd.read_csv(path)
    df['bot_x'] = df['Bot_Pos'].apply(lambda x: int(x.split(',')[0].strip('()')))
    df['bot_y'] = df['Bot_Pos'].apply(lambda x: int(x.split(',')[1].strip('()')))
    df['crew_x'] = df['Crew_Pos'].apply(lambda x: int(x.split(',')[0].strip('()')))
    df['crew_y'] = df['Crew_Pos'].apply(lambda x: int(x.split(',')[1].strip('()')))
    df['alien_x'] = df['Alien_Pos'].apply(lambda x: int(x.split(',')[0].strip('()')))
    df['alien_y'] = df['Alien_Pos'].apply(lambda x: int(x.split(',')[1].strip('()')))
    df['Distance_from_bot_to_crew'] = abs(df['bot_x'] - df['crew_x']) + abs(df['bot_y'] - df['crew_y'])
    df['Distance_from_alien_to_crew'] = abs(df['alien_x'] - df['crew_x']) + abs(df['alien_y'] - df['crew_y'])

    df.drop(['crew_x', 'crew_y', 'bot_x', 'bot_y','alien_x','alien_y'], axis=1, inplace=True)
    df['Bot_Move'] = df['Bot_Pos'].shift(-1)
    df['Bot_Move'] = df.apply(lambda row: clean_Bot_Move(row, df), axis=1)
    df =df.dropna()


    df["Bot_Cell_Encoded"] = df.apply(lambda row: map_coordinates_to_integer(row,"Bot_Pos"), axis=1)
    df["Crew_Cell_Encoded"] = df.apply(lambda row: map_coordinates_to_integer(row,"Crew_Pos"), axis=1)
    df["Bot_Move_Encoded"] = df.apply(lambda row: map_coordinates_to_integer(row,"Bot_Move"), axis=1)
    df["Alien_Pos_Encoded"] = df.apply(lambda row: map_coordinates_to_integer(row,"Alien_Pos"), axis=1)

    df['Closed_Cells_Encoded'] = df['Closed_Cells'].apply(encode_list_of_tuples)
    df['Closed_Cells_Encoded'] = closed_cells_encoder.fit_transform(df['Closed_Cells_Encoded'])
    df['Walls_Encoded'] = df['Wall_Cells'].apply(encode_list_of_tuples)
    df['Walls_Encoded'] = walls_encoder.fit_transform(df['Walls_Encoded'])

    label_encoded_df = df.copy()

    
    label_encoded_df = label_encoded_df.drop('Bot_Pos',axis =1)
    label_encoded_df = label_encoded_df.drop('Crew_Pos',axis =1)
    label_encoded_df = label_encoded_df.drop('Alien_Pos',axis =1)
    label_encoded_df = label_encoded_df.drop('Bot_Move',axis =1)
    label_encoded_df = label_encoded_df.drop('Closed_Cells',axis =1)
    label_encoded_df = label_encoded_df.drop('Wall_Cells',axis =1)



    return label_encoded_df

def train_data(data):
    final_data = data.copy()
    final_data = final_data.dropna()
    X = final_data.drop('Bot_Move_Encoded', axis=1)
    y = final_data['Bot_Move_Encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def Decision_Tree_Regressor(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, X_train,model

def reg_metrics(y_test, y_pred, X_train):
    from sklearn.metrics import mean_squared_error, r2_score 

    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    n = y_pred.shape[0]
    k = X_train.shape[1]
    adj_r_sq = 1 - (1 - r2)*(n-1)/(n-1-k)
    print("rmse:", rmse)
    print("r2:", r2)
    print("adj_r_sq:", adj_r_sq)

def create_model():
    data  = process_data()
    X_train, X_test, y_train, y_test = train_data(data)
    y_test, y_pred, X_train,model = Decision_Tree_Regressor(X_train, X_test, y_train, y_test)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)
    return model

def predict_model(model,input):
    Xnew = input
    ynew = model.predict(Xnew)
    return int(ynew)

model = create_model()

class LEARN_CONFIG(AI_Proj3.SHIP):
    def __init__(self, model, is_import = False):
        super(LEARN_CONFIG, self).__init__(is_import)
        self.model = model
        #print("closed = ",self.closed_cells)
        self.closed_encoded = encode_list_of_tuples(self.closed_cells)
        #print(self.closed_encoded)
        self.closed_encoded = closed_cells_encoder.fit_transform([self.closed_encoded])
        #print(self.wall_cells)
        #print("walls = ",self.wall_cells)
        self.walls_encoded = encode_list_of_tuples(self.wall_cells)
        #print(self.walls_encoded)
        self.walls_encoded = walls_encoder.fit_transform([self.walls_encoded])

    def visualize_grid(self, is_end = True): #pynb doesn't update the same image...
        if (not VISUALIZE) or (VISUALIZE and (not (self.ship.get_state(self.local_bot_pos) & BOT_CELL))):
            return

        data = []
        for i in range(self.ship.size):
            inner_list = []
            for j in range(self.ship.size):
                inner_list.append(self.ship.get_state((i, j)))

            data.append(inner_list)
        self.fig, self.ax = pyplot.subplots()
        self.image = pyplot.imshow(data, cmap='tab20')
        self.init_plots = True

        self.image.set_data(data)
        self.fig.canvas.draw_idle()
        pyplot.pause(.5)

        if is_end:
            pyplot.close(self.fig)


class LEARN_BOT(AI_Proj3.BOT_CONFIG):
    def __init__(self, ship):
        super(LEARN_BOT, self).__init__(ship)
        self.ship = ship
    
    def move_bot(self):
        bot_pos = self.local_bot_pos
        alien_pos = self.alien_pos
        #print("bot_pos",bot_pos)
        crew_pos = self.local_crew_pos
        crew_pos_encoded = crew_pos[0] * self.ship.size + crew_pos[1] + 1
        bot_pos_encoded = bot_pos[0] * self.ship.size + bot_pos[1] + 1
        alien_pos_encoded = alien_pos[0] * self.ship.size + alien_pos[1] + 1
        distance = AI_Proj3.get_manhattan_distance(bot_pos, crew_pos)
        distance = AI_Proj3.get_manhattan_distance(alien_pos,crew_pos)
        #print("bot",bot_pos_encoded)
        #print("crew",crew_pos_encoded)
        
        X_input = [[distance, bot_pos_encoded, crew_pos_encoded, self.ship.closed_encoded[0], self.ship.walls_encoded[0]],alien_pos_encoded]
        op = predict_model(self.ship.model, X_input)
        #print("OP",op)
        bot_x = (op - 1) % 11
        bot_y = (op - 1) // 11
        next_pos = (bot_y, bot_x)
        #print("bot_next move",next_pos)
        if 0 <= next_pos[0] < self.ship.size and 0 <= next_pos[1] < self.ship.size:
            for move in self.local_all_moves:
                bot_move = (bot_pos[0] + move[0], bot_pos[1] + move[1])
                #print("next_move", bot_move, next_pos)
                if bot_move == next_pos:
                    state = self.ship.get_state(next_pos)
                    self.action_result = state
                    if state != AI_Proj3.CLOSED_CELL and state != AI_Proj3.CREW_CELL:
                        #print(next_pos)
                        self.make_bot_move(next_pos)
                        return

        else:
            print("fail")

def print_data(detail):
    print("%18s %18s %18s %18s %18s %18s %18s %18s %18s %18s" % ("Avg Suc Moves", "Success Rate", "Min Suc. Moves", "Max Suc. Moves", "Avg Caught Moves", "Caught Rate", "Avg Fail Moves", "Failure Rate", "Avg Bot Crew Dist", "Crew Teleport Dist"))
    print(("%18s %18s %18s %18s %18s %18s %18s %18s %18s %18s" % (detail.s_moves, detail.success, detail.min_success, detail.max_success, detail.c_moves, detail.caught, detail.f_moves, detail.failure, detail.distance, detail.dest_dist)))

AI_Proj3.VISUALIZE=True
ship = LEARN_CONFIG(model)
avg_moves = run_simulations.DETAILS()
for _ in range(1000):
    moves, result = LEARN_BOT(ship).start_rescue()
    if result:
        avg_moves.update_min_max(moves)
        avg_moves.s_moves += moves
        avg_moves.success += 1
    else:
        avg_moves.f_moves += moves
        avg_moves.failure += 1

    avg_moves.distance += AI_Proj3.get_manhattan_distance(ship.bot_pos, ship.crew_pos)
    avg_moves.dest_dist += AI_Proj3.get_manhattan_distance(ship.crew_pos, ship.teleport_cell)
    ship.reset_positions()
avg_moves.get_avg(1000)
print_data(avg_moves)
        
        

