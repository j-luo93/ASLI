import csv
import os
import argparse

import random
random.seed(42) # fix seed for testing purposes

# code to parse the ielex data and build a dataset from an ancestral language and its descendants

default_dataset_path = 'data/ielex.tsv' # Indo-European dataset

# the header we will use in our custom .tsv data files; these are the properties each word will have. The two new columns, 'parsed_tokens' and 'split', will be populated with data we compute
header = ['language', 'iso_code', 'gloss', 'global_id', 
    'local_id', 'transcription', 'cognate_class', 'tokens', 'notes',
    'parsed_tokens', 'split']

# non: Old Norse, isl: Icelandic, fao: Faroese, swe: Swedish, dan: Danish, nor: Riksmal Norwegian, qov: Elfdalian
old_norse_iso_code = 'non'
nordic_iso_codes = {'isl', 'fao', 'swe', 'dan', 'nor', 'qov'}
# the ielex data has some language varieties, arguably dialects, that aren't assigned iso codes but do have 'language' entries
nordic_dialect_names = {'DANISH_FJOLDE', 'STAVANGERSK', 'GUTNISH_LAU', 'OLD_SWEDISH'}

latin_iso_code = 'lat'
# lat: Latin, spa: Spanish, por: Portuguese, fra: French, ita: Italian
romance_iso_codes = {'spa', 'por', 'fra', 'ita'}


def assign_iso_codes(languages, dataset_path=default_dataset_path):
    '''
    takes a .tsv dataset file where some languages are missing iso_code values \
        and saves a new .tsv dataset file where those languages now have iso_codes.
        Returns the path to the new dataset file as a str.
    
    languages: set of str, the 'language' values for the languages without iso codes
    '''
    file_path, file_name = os.path.split(dataset_path)
    new_filename = 'updated_' + file_name
    new_path = os.path.join(file_path, new_filename)

    with open(dataset_path) as f, open(new_path, 'w') as w:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        writer = csv.writer(w, delimiter='\t')
        writer.writerow(header)

        for line in reader:
            language = line[0]
            iso_code = line[1]
            if language in languages and iso_code=='':
                line[1] = 'q_' + language.lower()
                # the naming convention of q_ was chosen since iso codes qaa-qtz are 'reserved for local use'
            writer.writerow(line)

    return new_path


def filter_subfamily(parent_iso_code, daughter_iso_codes, dataset_path=default_dataset_path):
    '''
    returns a dictionary of the form {lang_iso_code: {global_id: (parent_line, daughter_line)}}, \
        which maps a daughter lang's iso code to a dict containing all cognate pairs in the dataset
        between the ancestor and that daughter lang.

    parent_iso_code: str, the iso code of the common ancestral language
    daughter_iso_codes: a set of str, the iso codes of the daughter languages of that parent
    dataset_path: str, the path to the .tsv dataset to be read from. Default written above.
    '''

    with open(dataset_path) as f:
        reader = csv.reader(f, delimiter='\t') # the file is a tsv (tab separated values)
        next(reader) # burn the header row

        # I make the assumption that a given language only has one word per particular cognate class. Running tests suggests that, at least for ielex.tsv, this assumption is valid.
        parent_dict = {} # cognates in the parent lang. {global_id: {cognate class: line}}
        daughter_dict = {} # cognates in the daughter langs. {global_id: {cognate class: {lang_iso_code: line}}}

        # go through the file once and populate the above dicts
        for line in reader:
            # these are all the different entires a line could have
            # language = line[0]
            iso_code = line[1]
            # gloss = line[2]
            global_id = line[3]
            # local_id = line[4]
            # transcription = line[5]
            cognate_class = line[6]
            # tokens = line[7]
            # notes = line[8]

            # note that we are making the assumption that there is a one-to-one mapping between languages as iso codes. This assumption seems to be true of the ielex data but can fall apart if multiple named doculects with different 'language' values are grouped under the same iso code.
            if iso_code == parent_iso_code:
                if global_id not in parent_dict:
                    parent_dict[global_id] = {}
                if cognate_class not in parent_dict[global_id]:
                    parent_dict[global_id][cognate_class] = line
            elif iso_code in daughter_iso_codes:
                if global_id not in daughter_dict:
                    daughter_dict[global_id] = {}
                if cognate_class not in daughter_dict[global_id]:
                    daughter_dict[global_id][cognate_class] = {}
                daughter_dict[global_id][cognate_class][iso_code] = line

    cognate_pair_dicts = {iso_code: {} for iso_code in daughter_iso_codes} # map a language to a dictionary with particular cognates {lang_iso_code: {global_id: (parent_line, daughter_line)}}

    # identify cognates where they exist both in the parent lang and at least one daughter lang
    for global_id in parent_dict:
        if global_id in daughter_dict:
            for cognate_class in parent_dict[global_id]:
                if cognate_class in daughter_dict[global_id]:
                    parent_line = parent_dict[global_id][cognate_class]

                    for lang_iso_code in daughter_dict[global_id][cognate_class]:
                        daughter_line = daughter_dict[global_id][cognate_class][lang_iso_code]
                        # make full copies of the parent and daughter lines instead of just setting a pointer to the same line. If all the separate daughter datasets share the same object for each parent line, then it's impossible to perform dataset-specific modifications of the parent lines, such as creating dataset splits.
                        cognate_pair_dicts[lang_iso_code][global_id] = (parent_line[:], daughter_line[:])
    
    return cognate_pair_dicts


def filter_daughter(parent_iso_code, daughter_iso_code, dataset_path=default_dataset_path):
    '''
    Filters a dataset to return a cognate dictionary only containing cognates that exist \
        in both the parent and the daughter language.
        Implemented as a subcase of filter_subfamily().

    parent_iso_code: str, the iso code of the parent language
    daughter_iso_code: str, the iso code of the daughter language
    dataset_path: str, the path to the .tsv file data will be read from. Default set above.

    returns: {global_id: (parent_line, daughter_line)}
    '''
    return filter_subfamily(parent_iso_code, {daughter_iso_code}, dataset_path=dataset_path)[daughter_iso_code]


def process_dataset(cognate_pair_dict):
    '''
    Processes a cognate pair dictionary, adding the new properties that we will be using \
        in performing our analysis and returning the new dictionary. Any new properties
        we add must also be named above in header as new columns.
    '''
    # I aim to modify the dictionary in-place, as this preserves the internal order. This is useful for making it such that casting the dict to an iterator still causes it to return items in the same order.
    # note(Derek): There's a more efficient way to do this that would only require looping through once overall instead of once per new column, but we don't have those speed concerns yet with the current dataset size.

    # add the parsed tokens column
    def append_parsed_tokens(line):
        iso_code = line[1]
        tokens = line[7]
        parsed_tokens = parse_tokens(tokens, iso_code)
        line.append(parsed_tokens)

    for global_id, lines in cognate_pair_dict.items():
        parent_line, daughter_line = lines
        append_parsed_tokens(parent_line)
        append_parsed_tokens(daughter_line)

    # add the dataset split column
    n_lines = len(cognate_pair_dict)
    cognate_ids = list(cognate_pair_dict.keys())
    # the split will be 80% training, 20% testing, and the training is 5-fold split for 5-fold validation
    # in other words, the fractions are 5 sets of 4/25 and one set of 5/25 
    random.shuffle(cognate_ids)

    split_factor = int(round(n_lines * (4/25)))

    # create five splits of 4/25ths of the data each
    for i in range(0, 5):
        for cog_id in cognate_ids[i * split_factor: (i+1) * split_factor]:
            parent_line, daughter_line = cognate_pair_dict[cog_id]
            parent_line.append(str(i+1))
            daughter_line.append(str(i+1))
    
    # the remaining 5/25ths goes to the test set
    for cog_id in cognate_ids[5 * split_factor:]:
        parent_line, daughter_line = cognate_pair_dict[cog_id]
        parent_line.append('test')
        daughter_line.append('test')

    return cognate_pair_dict


# an affricate map maps the stop part of an affricate to the fricative part
general_affricate_map = {'t': 'ʃ', 'd': 'ʒ'} # for broad cross-linguistic purposes
italian_affricate_map = general_affricate_map # geminated consonants can also create affricates, but we currently don't handle that

affricate_maps = {'general': general_affricate_map, 
    'ita': italian_affricate_map} # maps an iso code to that language's affricate map

def parse_tokens(word, language='general'):
    '''
    Processes a tokenized word to incorporate language-specific phonemic groupings.
    Specifically, the ielex data appears to have been tokenized in a way that \
        affricates are split into a separate stop-fricative pair, and we believe
        the model would be more accurate if it treated the fricative as a single segment.
    
    word: str, a tokenized word as presented in the ielex data. Tokens are space-separated.
    language: str, the iso code of the language that the word comes from, which affects \
        how affricates will be recognized. Default is a very generic cross-linguistic
        affricate detector. 

    returns: str, the retokenized word, tokens are space-separated
    '''
    if language in affricate_maps:
        affricate_map = affricate_maps[language]
    else:
        affricate_map = affricate_maps['general']

    tokens = word.split(' ')
    new_tokens = []
    
    prev_token = None
    for token in tokens:
        if prev_token in affricate_map and affricate_map[prev_token] == token:
            # rewrite the last entry in new_tokens
            new_tokens[-1] = prev_token + token
        else:
            new_tokens.append(token)
        prev_token = token
    
    return str.join(' ', new_tokens)


# save the custom datasets to their own files. For each daughter lang, we split its cognate pair dict into two files: one for the parent lang, one for the daughter lang.
def save_dataset(cognate_pair_dict, output_dir=None):
    '''
    Saves a cognate pair dict for a particular daughter language as two .tsv files, \
        one containing the parent lang's cognates, the other the daughter's.

    cognate_pair_dict: the cognate pair dictionary for the daughter language, formatted like one of the dictionaries in the output of filter_subfamily()
    output_dir: directory the files will be saved in, under /data/. Default folder name is 'ParentLang-DaughterLang Cognates'
    '''

    if len(cognate_pair_dict) == 0: # must have something to save
        return

    # we extract an arbitrary line from the dictionary to obtain the parent and daughter iso codes
    parent_line, daughter_line = next(iter(cognate_pair_dict.values())) # popping from an iterator is best way of doing this: it uses the least space since it doesn't load the whole dictionary

    # this function can fill in columns that are empty with blank data, but it can't discard columns if the header is underspecified
    assert len(header) >= len(parent_line) == len(daughter_line)
    # if the cognate dictionary hasn't been processed to populate the new column values for each cognate, we must append blank entries to the lines so that the resulting .tsv has entries in all columns
    # note that I assume any new column properties we add are added to the end of the header, on the right side
    n_missing_columns = len(header) - len(parent_line)
    pad = [''] * n_missing_columns

    parent_iso, daughter_iso = parent_line[1], daughter_line[1]

    if output_dir is None: # generate default directory name by joining the iso codes
        output_dir = parent_iso + '-' + daughter_iso
    
    output_dir = os.path.join('data', output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    parent_file_path = os.path.join(output_dir, parent_iso + '.tsv')
    daughter_file_path = os.path.join(output_dir, daughter_iso + '.tsv')

    with open(parent_file_path, 'w') as f_p, open(daughter_file_path, 'w') as f_d:
        writer_p = csv.writer(f_p, delimiter='\t')
        writer_d = csv.writer(f_d, delimiter='\t')

        writer_p.writerow(header)
        writer_d.writerow(header)

        for k in cognate_pair_dict.keys():
            parent_line, daughter_line = cognate_pair_dict[k]
            # the columns are padded with blank entries if they are missing any data
            writer_p.writerow(parent_line + pad)
            writer_d.writerow(daughter_line + pad)


def read_saved_dataset(data_path, parent_iso, daughter_iso):
    '''
    Reads two paired .tsv datasets, like those created by save_dataset, \
        and recreates their cognate dictionary
    
    data_path: str, path to the folder the two .tsv are kept in
    parent_iso: str, iso code of the parent language
    daughter_iso: str, iso code of the daughter language
    '''
    parent_path = os.path.join(data_path, parent_iso + '.tsv')
    daughter_path = os.path.join(data_path, daughter_iso + '.tsv')

    cog_dict = {}

    with open(parent_path) as f_p, open(daughter_path) as f_d:
        reader_p = csv.reader(f_p, delimiter='\t')
        reader_d = csv.reader(f_d, delimiter='\t')

        next(reader_p) # burn the header rows
        next(reader_d)

        for line_tuple in zip(reader_p, reader_d):
            cognate_id = line_tuple[0][3] # parent and daughter lines should have identical cognate_ids
            cog_dict[cognate_id] = line_tuple

    return cog_dict


def cog_dict_to_splits(cognate_dict):
    '''
    Converts a cognate dictionary to a dictionary of the separate data splits and shuffles the splits.
    '''
    split_dict = {} # will map a name of a given split to a list of cognate tuples, which are paired cognate lines in the parent and daughter languages

    for line_tuple in cognate_dict.values():
        split = line_tuple[0][10] # the parent and daughter belong to the same split
        if split not in split_dict:
            split_dict[split] = []
        split_dict[split].append(line_tuple)
    
    for split_list in split_dict.values():
        random.shuffle(split_list)

    return split_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process .tsv cognate datasets")

    parser.add_argument('--data_path', default='data/ielex.tsv', help="path to .tsv dataset")
    parser.add_argument('--par_iso', required=True, help="iso code of the parent language")
    parser.add_argument('--dau_iso', required=False, nargs='+', help="iso code(s) of daughter language(s)")
    parser.add_argument('--dau_name', required=False, nargs='+', help="name(s) of daughter language(s) without iso codes")
    parser.add_argument('--preserve_data', action="store_true", help="include to prevent data processing")
    parser.add_argument('-v', '--verbose', action="store_true", help="increase output verbosity")

    args = parser.parse_args()

    if args.dau_iso is None and args.dau_name is None:
        parser.error("at least one of --dau_iso and --dau_name is required")

    args.dau_iso = set(args.dau_iso) if args.dau_iso else set()
    
    if args.dau_name:
        args.dau_name = set(args.dau_name)
        # process the dataset to add iso codes for the daughter languages without iso codes
        new_path = assign_iso_codes(args.dau_name, args.data_path)
        # add their new iso codes to the list of daughter iso codes
        args.dau_iso = args.dau_iso | {'q_' + lang.lower() for lang in args.dau_name}
        if args.verbose:
            print('added iso codes in', args.data_path, 'for', str.join(', ', args.dau_name))
            print('now reading', new_path)
        args.data_path = new_path

    cognate_dicts = filter_subfamily(args.par_iso, args.dau_iso, dataset_path=args.data_path)
    
    if args.verbose:
        print('reading', args.data_path, 'for', args.par_iso, 'and', str.join(', ', args.dau_iso))
        print("# of identified cognate pairs per language:")
        for iso, cog_dict in cognate_dicts.items():
            print('\t' + iso, ':', len(cog_dict))
    
    if not args.preserve_data:
        for iso, cog_dict in cognate_dicts.items():
            cognate_dicts[iso] = process_dataset(cog_dict)
        if args.verbose:
            print("retokenized cognates and added dataset splits")
    else:
        if args.verbose:
            print("preserving data, no edits applied")
    
    for cog_dict in cognate_dicts.values():
        save_dataset(cog_dict)
    if args.verbose:
        print("saved dataset(s) under data")
