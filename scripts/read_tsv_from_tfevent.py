import argparse
import csv
import glob
import os
from pathlib import Path

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
    return map(str, Path(directory).glob('**/event*'))
    # return glob.glob(os.path.join(directory, '*', '*', 'event*'))


def read_event_file(out_folder, path):
    header = ['step', 'tag', 'value']
    # what is represented by 'value' depends on the tag — it could be loss, or accuracy, or the gradient norm
    
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    # this loses info about the full path; alternative below
    # file_name = path.replace('/', '_') + '.tsv' # preserves all info about the full path but creates very long filenames
    # maybe a smarter alternative would be just to have a two-line header, with the first line being the full filepath. This is slightly unconventional but it's smart here.
    # NOTE(j_luo) I'm going with the long file name to perserve more information.
    name = path.replace("/", '__')
    file_name = (out_folder / name).with_suffix('.tsv')

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

    parser.add_argument('--data_path', required=True,
                        help='date-level directory to export tfevents files from (eg log/2020-08-18)')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser.add_argument('--out_folder', type=str, default='processed_log', help='Output folder for the processed files.')

    args = parser.parse_args()

    paths = get_event_file_paths(args.data_path)
    if args.verbose:
        print("Reading the following tfevent files:")
    for path in paths:
        if args.verbose:
            print('\t' + path)
        read_event_file(args.out_folder, path)
    if args.verbose:
        print(f"Saved .tsv files under {args.out_folder}.")
