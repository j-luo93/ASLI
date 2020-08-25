# script to read a bunch of log files in a date-level directory

import os
import glob
import csv
import argparse

# currently hardcoded to care only about model encoder type, the config, dropout, and embedding size. There may be some more elegant way to create modularity but this will do for now.

def get_log_file_paths(directory):
    '''returns filepaths to all log files within a date-level directory'''
    return glob.glob(os.path.join(directory, '*', '*', 'log'))

def read_log_files(paths):
    '''
    Read all the specified log files and return a datastructure representing the runs with the following structure:

    a dictionary that maps a tuple of strs (char_emb_size, hidden_size, dropout, model_encoder_type, config), which uniquely identify a run, to a dictionary that maps the name of an evaluated property to its mean value at 10,000 steps.
    '''
    # these entries are all strs for convenience of reading files/avoid floating point comparison nonsense with dropout
    emb_sizes = hidden_sizes = {'110', '220', '440', '128', '256', '512'} # this double equality may change in the future
    dropout_sizes = {'0.0', '0.2', '0.4', '0.6'}
    enc_types = {'lstm', 'cnn'}
    config = {'ZSLatIta', 'ZSLatItaPhono'}

    data_dict = {}
    
    for path in paths:
        run_info = [''] * 5 # will cast to tuple later to use as key
        run_dict = {}

        with open(path) as f:
            for line in f:
                if 'char_emb_size:' in line:
                    run_info[0] = line.strip(' \n').split(': ')[1]
                if 'hidden_size:' in line:
                    run_info[1] = line.strip(' \n').split(': ')[1]
                if 'dropout:' in line:
                    run_info[2] = line.strip(' \n').split(': ')[1]
                if 'model_encoder_type:' in line:
                    run_info[3] = line.strip(' \n').split(': ')[1]
                if 'config:' in line:
                    run_info[4] = line.strip(' \n').split(': ')[1]


                # I assume the model has only been evaluated once, so any instances of 'eval/' signify the Eval 10000 block [the final evaluation]
                if 'eval/' in line:
                    eval_splits = line.strip(' \n').split('|')
                    name = eval_splits[1]
                    value = eval_splits[4]
                    run_dict[name] = value

        assert tuple(run_info) not in data_dict
        data_dict[tuple(run_info)] = run_dict
    
    return data_dict


def write_to_csv(data_dict):
    '''
    Reads a data_dict as produced by read_log_files() and saves relevant csvs

    Again, all hardcoded at the moment
    '''
    enc_type_options = {'cnn', 'lstm'}
    config_options = {'ZSLatIta', 'ZSLatItaPhono'}
    lang_options = {'ita', 'cat', 'spa', 'ron', 'por'}
    prec_n_options = {'p@1', 'p@5'}

    dropout_options = ['0.0', '0.2', '0.4', '0.6']
    emb_hid_size_options = ['110', '220', '440', '128', '256', '512']

    # maps each language to a dict that maps a tuple (enc_type, config, p@N) to a 2d-array of values. Dimension 0 corresponds to dropout (0.0, 0.2, 0.4, 0.6) and Dimension 1 corresponds to emb/hidden size (which for now at least are assumed to be equal): (110, 220, 440, 128, 256, 512). So for instance, entry [3][1] would be an experiment with dropout=0.6 and emb/hidden size 220.
    csv_dict = {}
    for lang in lang_options:
        csv_dict[lang] = {}
        for enc_type in enc_type_options:
            for config in config_options:
                for prec_n in prec_n_options:
                    key = (enc_type, config, prec_n)
                    assert key not in csv_dict[lang]
                    csv_dict[lang][key] = [
                        ['','','','','',''],
                        ['','','','','',''],
                        ['','','','','',''],
                        ['','','','','',''],
                    ] # for some reason using the [['']*6]*4 shorthand won't work, as it actually creates copies of the same object and they share reference

    for run_tup, run_dict in data_dict.items():
        emb_size = run_tup[0]
        dropout = run_tup[2]
        enc_type = run_tup[3]
        config = run_tup[4]

        for name, value in run_dict.items():
            if 'test' in name: # we only care about test runs
                splits = name.strip(' \n').split('@')
                lang = splits[1][:3] # super hacky but correct
                prec_n = 'p@' + splits[2]

                key = (enc_type, config, prec_n)

                dropout_id = dropout_options.index(dropout)
                emb_size_id = emb_hid_size_options.index(emb_size)
                assert csv_dict[lang][key][dropout_id][emb_size_id] == ''
                csv_dict[lang][key][dropout_id][emb_size_id] = value
    
    if not os.path.exists('processed_log'):
        os.mkdir('processed_log')

    for lang in lang_options:
        file_name = os.path.join('processed_log', 'log_to_csv' + lang + '.csv')
        with open(file_name, 'w') as f:
            writer = csv.writer(f)

            for tup, val_array in csv_dict[lang].items():
                identifier = ' '.join(tup) # goes in the 0,0 value, which is empty
                header = [identifier] + emb_hid_size_options # technically not a header since we have several of them
                writer.writerow(header)

                for i, row in enumerate(val_array):
                    writer.writerow([dropout_options[i]] + row)
                
                # write a couple blank rows for space between entries
                writer.writerow([''] * 6)
                writer.writerow([''] * 6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process log files and compile results")

    parser.add_argument('--data_path', required=True, help='path to date-level folder to compile log files from')

    args = parser.parse_args()

    paths = get_log_file_paths(args.data_path)
    write_to_csv(read_log_files(paths))