import pandas as pd
import numpy as np
import os
import glob

def main(dirs_filename, data_filename, epoch_precision, index_name):
    """Run main script"""

    # read in the input files
    dirs = pd.read_csv(dirs_filename)    
    dirs = dirs[dirs["used_"] == True]
    filenames = pd.read_csv(data_filename)    
    filenames = filenames[filenames["used_"] == True]

    # merge each file according to date
    for dir_iter, dir in dirs.iterrows():
        for row_iter, row in filenames.iterrows():

            # read and sort the filenames by date
            all_files = glob.glob(os.path.join(dir["dirs"], row["filenames"]))
            all_files.sort(key=os.path.getmtime)
            all_df = []

            # parse and trim each of the files
            epoch_trim_cummulative = 1
            epoch_trim = 0
            for f_iter, f in enumerate(all_files):
                df = pd.read_csv(f, dtype=np.float32)
                df['file'] = f.split('/')[-1] # add in the filename to the dataframe

                # trim at the nearest epoch_percent
                if f_iter < len(all_files) - 1:
                    epoch_last = df.tail(1)[index_name].to_numpy()[0]
                    epoch_trim = np.floor(epoch_last / epoch_precision) * epoch_precision
                    df = df[df[index_name]<=epoch_trim]

                # update the epoch
                if f_iter != 0:
                    df[index_name] = df[index_name] + epoch_trim_cummulative

                all_df.append(df)
                epoch_trim_cummulative += epoch_trim;

            # concatenate and write to disk
            merged_df = pd.concat(all_df, ignore_index=True, sort=True)
            merged_df.to_csv(os.path.join(dir["dirs"], row["merged_names"]))

# Run main
if __name__ == "__main__":
    # Input files
    path = ""
    dirs_filename = path + "MergeTrainTestRunsDirs.csv"
    data_filename = path + "MergeTrainTestRunsFilenames.csv"
    epoch_precision = 1000
    index_name = "Epoch"

    # Name of the index
    main(dirs_filename, data_filename, epoch_precision, index_name)