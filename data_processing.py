import pandas as pd
import csv

# code to filter the ielex data to only include Old Norse and descendants and build a custom dataset

dataset_path = 'data/ielex.tsv' # hardcoded — we want to get the above Old Norse family languages

with open(dataset_path) as f:
    reader = csv.reader(f, delimiter='\t') # the file is a tsv (tab separated values)
    next(reader) # burn the header row

    for line in reader:
        # these are all the different entires a line could have
        language = line[0]
        iso_code = line[1]
        gloss = line[2]
        global_id = line[3]
        local_id = line[4]
        transcription = line[5]
        cognate_class = line[6]
        tokens = line[7]
        notes = line[8]
