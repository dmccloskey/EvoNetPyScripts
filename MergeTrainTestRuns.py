import pandas as pd
import os
import glob

def main(data_dir, data_filename, epoch_precision, index_name):
    """Run main script"""

    # read in the input files
    filenames = pd.read_csv(data_filename)    
    filenames = filenames[filenames["used_"] == True]

    # merge each file according to date
    for row_iter, row in filenames.iterrows():

        # read and sort the filenames by date
        all_files = glob.glob(os.path.join(data_dir, row["filename"]))
        all_files.sort(key=os.path.getmtime)
        all_df = []

        # parse and trim each of the files
        for f in all_files:
            df = pd.read_csv(f, sep=',')
            df['file'] = f.split('/')[-1] # add in the filename to the dataframe            
            epoch_trim = df.tail(1)[index_name] % epoch_precision # trim at the nearest epoch_percent
            df = df[df[index_name]<=epoch_trim]
            all_df.append(df)

        # concatenate and write to disk
        merged_df = pd.concat(all_df, ignore_index=True, sort=True)
        df_merged.to_csv(row["filename"])

# Run main
if __name__ == "__main__":
    # Input files
    data_dir = "C:/Users/dmccloskey/Desktop/stickExample/"
    data_filename = data_dir + "MergeTrainTestRunsInput.csv"
    epoch_precision = 1000
    index_name = "Epoch"

    # Name of the index
    main(data_dir, data_filename, epoch_precision, index_name)