import pandas as pd
import csv

# code to parse the ielex data and build a dataset from an ancestral language and its descendants

dataset_path = 'data/ielex.tsv'

old_norse_iso_code = 'non'
# non: Old Norse, isl: Icelandic, fao: Faroese, swe: Swedish, dan: Danish, nor: Norwegian
nordic_iso_codes = {'isl', 'fao', 'swe', 'dan', 'nor'}

latin_iso_code = 'lat'
# lat: Latin, spa: Spanish, por: Portuguese, fra: French, ita: Italian
romance_iso_codes = {'spa', 'por', 'fra', 'ita'}

def filter_subfamily(parent_lang_iso, daughter_iso_codes):
    '''
    parent_lang_iso: str, the iso code of the common ancestral language
    daughter_iso_codes: a set of str, the iso codes of the daughter languages of that parent

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

            if iso_code == parent_lang_iso:
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
        else:
            print('the straggler is', global_id)
    
    return cognate_pair_dicts


romance_cognate_pair_dicts = filter_subfamily(latin_iso_code, romance_iso_codes)

# for each daughter lang, count the number of attested cognates. We'll pick the top two, training the model on parent -> daughter_1 and benchmark performance on parent -> daughter_2
for lang in romance_cognate_pair_dicts:
    print(lang, ':', len(romance_cognate_pair_dicts[lang]))


# sample some data to examine
for k in list(romance_cognate_pair_dicts['ita'])[:10]:
    print(romance_cognate_pair_dicts['ita'][k])

# TODO write new prepared data to file


