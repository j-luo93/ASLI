import glob
import os
import csv
import argparse

from tensorflow.python.summary.summary_iterator import summary_iterator

# the summary_iterator returns objects with the following form:
# wall_time: float
# step: int
# summary {
#   value {
#       tag: str
#       simple_value: float
#   }
# }

def get_event_file_paths(directory):
    '''returns filepaths to all tfevents files in a certain directory'''
    return glob.glob(os.path.join(directory, '*', '*', 'event*'))

def read_event_file(path):
    header = ['step', 'tag', 'score']

    if not os.path.exists('processed_log'):
        os.mkdir('processed_log')
    file_name = os.path.join('processed_log', os.path.basename(path) + '.tsv') # this loses info about the full path; alternative below
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="export a tfevents file to a tsv")

    parser.add_argument('--data_path', required=True, help='date-level directory to export tfevents files from (eg log/2020-08-18)')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')

    args = parser.parse_args()

    paths = get_event_file_paths(args.data_path)
    if args.verbose:
        print("Reading the following tfevent files:")
    for path in paths:
        if args.verbose:
            print('\t' + path)
        read_event_file(path)
    if args.verbose:
        print("Saved .tsv files under processed_log/")
