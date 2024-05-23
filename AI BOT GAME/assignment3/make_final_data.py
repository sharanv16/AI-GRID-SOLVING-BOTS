import pandas as pd
import os

is_bonus = True
is_general = False

# Define the folder path
if is_bonus:
    folder_path = 'general_bonus' if is_general else 'single_bonus'
else:
    folder_path = 'general' if is_general else 'single'
data_csv = '_general.csv' if is_general else 'single.csv'
layout_csv = '_layout.csv' if is_general else 'layout.csv'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter files that match the pattern "*_general.csv"
general_files = [f for f in files if f.endswith(data_csv)]
print(len(general_files))
# Iterate through each general file
for general_file in general_files:
    # Extract the corresponding layout file name
    layout_file = general_file.replace(data_csv, layout_csv)
    
    # Load the CSV files
    general_file_path = os.path.join(folder_path, general_file)
    layout_file_path = os.path.join(folder_path, layout_file)
    
    df_general = pd.read_csv(general_file_path)
    df_layout = pd.read_csv(layout_file_path)
    
    # Repeat the single row in df_layout to match the number of rows in df_general
    num_rows_to_repeat = len(df_general) - len(df_layout)
    if num_rows_to_repeat > 0:
        repeated_rows = pd.concat([df_layout] * num_rows_to_repeat, ignore_index=True)
        df_layout = pd.concat([df_layout, repeated_rows], ignore_index=True)
    
    # Combine the dataframes
    final_df = pd.concat([df_general, df_layout], axis=1)
    
    # Write the combined dataframe to a new CSV file
    final_csv_path = os.path.join(folder_path, 'final_data.csv')
    mode = 'a' if os.path.exists(final_csv_path) else 'w'
    final_df.to_csv(final_csv_path, mode=mode, index=False, header=not os.path.exists(final_csv_path))

print("Combined CSV file created successfully!")
