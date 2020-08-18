import glob
import os
import csv
import argparse

from tensorflow.python.summary.summary_iterator import summary_iterator

# print(next(summary_iterator('log/2020-08-17/ZSLatItaPhono/02-12-14/events.out.tfevents.1597644744.rosetta3.4547.0')))

# the summary_iterator returns objects with the following form:
# wall_time: float
# step: int
# summary {
#   value {
#       tag: str
#       simple_value: float
#   }
# }
sample_path = 'log/2020-08-17/ZSLatItaPhono/02-12-14/events.out.tfevents.1597644744.rosetta3.4547.0'
# for e in summary_iterator(sample_path):
#     if e.step == 10000:
#         print(e.summary.value)

def get_event_file_paths(directory):
    '''returns filepaths to all tfevents files in a certain directory'''
    return glob.glob(os.path.join(directory, '*', 'event*'))

def read_event_file(path):
    header = ['step', 'tag', 'score']

    file_name = os.path.basename(path) + '.tsv' # this loses info about the full path; alternative below
    # file_name = path.replace('/', '_') + '.tsv' # preserves all info about the full path but creates very long filenames
    # maybe a smarter alternative would be just to have a two-line header, with the first line being the full filepath. This is slightly unconventional but it's smart here.

    with open(file_name, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        
        writer.writerow(header)
        for e in summary_iterator(path):
            # there's only one entry in each Event, but this is the only way I have found to extract the Value
            for v in e.summary.value:
                line = [e.step, v.tag, v.simple_value]
                writer.writerow(line)

read_event_file(sample_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="export a tfevents file to a tsv")

    parser.add_argument('--data_path', required=True, help='folder to export tfevents files from')
    # parser.add_argument('--step', default=10000, nargs='?', help='step count to collect data from')
    parser.add_argument('--output_file', required=False)
    