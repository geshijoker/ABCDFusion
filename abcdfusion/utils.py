import os
import shutil
import time
import logging

def check_make_dir(path):
    """
    Check if a directory exists given the path, if not create one.
    """
    # You should change 'test' to your preferred folder.
    mydir = os.path.join('./', path)
    check_folder = os.path.isdir(mydir)

    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(mydir)
        print("created folder : ", mydir)
    else:
        print(mydir, "folder already exists.")
    return mydir

def clean_directory(path):
    """
    Clean the folders of files that should be disclosed in a linux style system
    """
    for root, dirs, files in os.walk(path):
        # Remove .DS_Store files
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")

        # Remove .ipynb_checkpoints directories
        for dir_name in dirs:
            if dir_name == ".ipynb_checkpoints":
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")

        # Remove __pycache__ directories
        for dir_name in dirs:
            if dir_name == "__pycache__":
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")

if __name__ == "__main__":
    target_directory = "./"  # Change this to the root directory where you want to start the cleanup
    clean_directory(target_directory)