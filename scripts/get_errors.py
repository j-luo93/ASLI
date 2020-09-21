import sys

import pandas as pd

if __name__ == "__main__":
    in_path = sys.argv[1]
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    number = int(sys.argv[3])

    df = pd.read_csv(in_path, sep='\t', keep_default_na=True)
    correct = df['correct@5']
    out_df = df[~correct].sample(n=number)
    out_df.to_csv(out_path, sep='\t', index=None)
