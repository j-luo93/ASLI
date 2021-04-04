# Benchmark Sound Law LSTM

Part of a project that tries to automatically derive sound laws from a list of cognates.

This project uses the ielex dataset as provided in [Jäger et al. 2017](https://www.aclweb.org/anthology/E17-1113.pdf), "Using support vector machines and state-of-the-art algorithms for phonetic alignment to ientify cognates in multi-lingual wordlists".

# Prepare data
* Obtain NorthEuraLex dataset by running `wget http://www.sfs.uni-tuebingen.de/~jdellert/northeuralex/0.9/northeuralex-0.9-forms.tsv`. 
* Obtain cognate set dataset and merge it with NorthEuraLex by using `wikt_reader` library. You would get a family file.
* Prepare input data by running
```
python scripts/process_data_wikt.py --data_path <path_to_family_file> --source <src> --targets <tgt_langs> --no_need_transcriber
```
For instance, for the Germanic language family, run
```
python scripts/process_data_wikt.py --data_path data/Germanic.tsv --source gem-pro --targets eng deu isl nor swe dan nld --no_need_transcriber
```
# Dependencies
* various packages in `requirements.txt`. Run `pip install -r requirements.txt`.
