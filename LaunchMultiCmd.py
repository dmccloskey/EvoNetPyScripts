from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from subprocess import Popen

def readCommandsCsv(filename):
    """Creates a pandas data frame with headers for 'commands'
    from a .csv file and returns each of the columns as seperate entities.  The column 'used_' is
    used to select what rows to use

    Args:
        filename: name of the file

    Returns:
        pandas data frame with headers for 'filenames','labels','colors','markers'
    """
    data = pd.read_csv(filename)
    data_used = data[data["used_"]==True]
    commands = list(data_used.loc[:, "commands"]) 
    return commands

def main(data_dir, filename):
    """Run main script"""

    # read in the input files
    commands = readCommandsCsv(filename)

    # Run in parallel
    n_threads = 16 #28
    thread_count = 0
    commands_to_run = []
    for command in commands:
        if thread_count < n_threads:
            commands_to_run.append(command)
            thread_count += 1
        if thread_count >= n_threads or command == commands[-1]:
            DETACHED_PROCESS = 0x00000008
            procs = [ Popen(i, creationflags=DETACHED_PROCESS, shell=True) for i in commands_to_run ]
            for p in procs:
                p.wait()
            commands_to_run = []
            thread_count = 0

# Run main
if __name__ == "__main__":
    # Input files
    data_dir = ""
    filename = data_dir + "CommandsToRun.csv"
    main(data_dir, filename)