import csv
import os

# code to parse the ielex data and build a dataset from an ancestral language and its descendants

default_dataset_path = 'data/ielex.tsv' # Indo-European dataset

old_norse_iso_code = 'non'
# non: Old Norse, isl: Icelandic, fao: Faroese, swe: Swedish, dan: Danish, nor: Norwegian
nordic_iso_codes = {'isl', 'fao', 'swe', 'dan', 'nor'}

latin_iso_code = 'lat'
# lat: Latin, spa: Spanish, por: Portuguese, fra: French, ita: Italian
romance_iso_codes = {'spa', 'por', 'fra', 'ita'}

def filter_subfamily(parent_iso_code, daughter_iso_codes, dataset_path=default_dataset_path):
    '''
    parent_iso_code: str, the iso code of the common ancestral language
    daughter_iso_codes: a set of str, the iso codes of the daughter languages of that parent
    dataset_path: str, the path to the .tsv dataset to be read from. Default written above.

    returns a dictionary of the form {lang_iso_code: {global_id: (parent_line, daughter_line)}}, \
        which maps a daughter lang's iso code to a dict containing all cognate pairs in the dataset
        between the ancestor and that daughter lang.
    '''

    with open(dataset_path) as f:
        reader = csv.reader(f, delimiter='\t') # the file is a tsv (tab separated values)
        next(reader) # burn the header row

        # I assume that any given language, parent or daughter, only has one word per particular cognate class. Running code on ielex.tsv, at least, it seems this assumption is correct.
        parent_dict = {} # cognates in the parent lang. {global_id: {cognate class: line}}
        daughter_dict = {} # cognates in the daughter langs. {global_id: {cognate class: {lang_iso_code: line}}}

        # go through the file once and populate the above dicts
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
            # notes = line[8]

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

    print('identified', len(parent_dict), 'parent cognates')
    print('identified', len(daughter_dict), 'daughter cognates')

    cognate_pair_dicts = {iso_code: {} for iso_code in daughter_iso_codes} # map a language to a dictionary with particular cognates {lang_iso_code: {global_id: (parent_line, daughter_line)}}

    # identify cognates where they exist both in the parent lang and at least one daughter lang
    for global_id in parent_dict:
        if global_id in daughter_dict:
            for cognate_class in parent_dict[global_id]:
                if cognate_class in daughter_dict[global_id]:
                    parent_line = parent_dict[global_id][cognate_class]

                    for lang_iso_code in daughter_dict[global_id][cognate_class]:
                        daughter_line = daughter_dict[global_id][cognate_class][lang_iso_code]
                        cognate_pair_dicts[lang_iso_code][global_id] = (parent_line, daughter_line)
    
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
    return filter_subfamily(parent_iso_code, {daughter_iso_code}, dataset_path)[daughter_iso_code]

# testing filter_subfamily
romance_cognate_pair_dicts = filter_subfamily(latin_iso_code, romance_iso_codes)

# for each daughter lang, count the number of attested cognates. We'll pick the top two, training the model on parent -> daughter_1 and benchmark performance on parent -> daughter_2
for lang in romance_cognate_pair_dicts:
    print(lang, ':', len(romance_cognate_pair_dicts[lang]))


# # sample some data to examine
# for k in list(romance_cognate_pair_dicts['ita'])[:10]:
#     print(romance_cognate_pair_dicts['ita'][k])

# save the custom datasets to their own files. For each daughter lang, we split its cognate pair dict into two files: one for the parent lang, one for the daughter lang.
def save_dataset(cognate_pair_dict, output_dir=None):
    '''
    Saves a cognate pair dict for a particular daughter language as two .tsv files, \
        one containing the parent lang's cognates, the other the daughter's.

    cognate_pair_dict: the cognate pair dictionary for the daughter language, formatted like one of the dictionaries in the output of filter_subfamily()
    output_dir: directory the files will be saved in, under /data/. Default folder name is 'ParentLang-DaughterLang Cognates'
    '''

    # we extract an arbitrary line from the dictionary to obtain the parent and daughter iso codes
    parent_line, daughter_line = next(iter(cognate_pair_dict.values())) # popping from an iterator is best way of doing this: it uses the least space since it doesn't load the whole dictionary

    if output_dir is None: # generate default directory name
        parent_language, daughter_language = parent_line[0], daughter_line[0]
        output_dir = parent_language + '-' + daughter_language + ' Cognates'
        output_dir = output_dir.title() # format to make the folder easier to read
    # note (Derek): it occurs to me that the above method of extracting the language names can potentially cause an undesirable bug. Since multiple distinct doculects could be under the same iso code but have a different listed language name, it is theoretically possible that the same cognate dictionary could be saved to a different directory on different runs, if the line that is popped from the iterator changes. If it is true that Python dictionaries have an internal order in python 3, however, then the line popped from the iterator is deterministic (for a given dataset) and this is not a problem.
    
    output_dir = os.path.join('data', output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    parent_iso, daughter_iso = parent_line[1], daughter_line[1]
    parent_file_path = os.path.join(output_dir, parent_iso + '.tsv')
    daughter_file_path = os.path.join(output_dir, daughter_iso + '.tsv')

    with open(parent_file_path, 'w') as f_p, open(daughter_file_path, 'w') as f_d:
        writer_p = csv.writer(f_p, delimiter='\t')
        writer_d = csv.writer(f_d, delimiter='\t')

        header = ['language', 'iso_code', 'gloss', 'global_id', 
            'local_id', 'transcription', 'cognate_class', 'tokens', 'notes']
        writer_p.writerow(header)
        writer_d.writerow(header)

        for k in cognate_pair_dict.keys():
            parent_line, daughter_line = cognate_pair_dict[k]
            writer_p.writerow(parent_line)
            writer_d.writerow(daughter_line)
    
save_dataset(romance_cognate_pair_dicts['fra'])

save_dataset(filter_daughter('lat', 'por'))