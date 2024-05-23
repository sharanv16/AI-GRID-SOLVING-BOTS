import os
import glob

def rename_files_with_wildcard(wildcard_pattern, new_name):
    files = sorted(glob.glob(wildcard_pattern))
    bonus_start = 64
    for itr, old_name in enumerate(files):
        _, extension = os.path.splitext(old_name)

        new_file_name = str(bonus_start + itr) + new_name + extension

        os.rename(old_name, new_file_name)
        print(f"Renamed '{old_name}' to '{new_file_name}'")

wildcard_pattern = "*_layout.csv"
new_name = "_layout"
rename_files_with_wildcard(wildcard_pattern, new_name)
