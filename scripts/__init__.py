import os

# Get the absolute path of the current file
current_file = os.path.abspath(__file__)

# Get the directory of the current file
current_dir = os.path.dirname(current_file)

# Get the parent directory of the current directory
base_dir = os.path.dirname(current_dir)