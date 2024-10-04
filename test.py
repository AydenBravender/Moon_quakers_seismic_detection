import numpy as np

import os

def remove_csv_files(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .csv extension
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                # Remove the CSV file
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

if __name__ == "__main__":
    # Replace with the path to your directory
    directory = "data"
    remove_csv_files(directory)
