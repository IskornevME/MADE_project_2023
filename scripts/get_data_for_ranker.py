"""
This module takes file with docs + gen text (for example 'data_for_metrics_rugpt_1000.json') and make tsv file for any ranker.
Usage: python get_data_for_ranker.py -i input_file_name.json -o output_file_name.tsv
"""

import sys, getopt
import json
import pandas as pd


def main(inputfile, outputfile):
    ifile = open(inputfile)
    data_rugpt = json.load(ifile)
    df_processed = pd.DataFrame.from_dict(data_rugpt)
    
    columns_titles = ["query", "is_selected", "passage_text"]
    df_processed = df_processed.reindex(columns=columns_titles)
    df_processed.rename(columns={'is_selected': 'label', 'passage_text': 'body'}, inplace=True)
    
    for row in df_processed.iterrows():
        tmp = row[1]["body"][-1]
        row[1]["body"][-1] = ''.join(tmp.splitlines())
        
    df_processed_exp = df_processed.explode(["label", "body"])
    df_processed_exp.dropna(inplace=True)
    df_processed_exp.to_csv(outputfile, sep="\t", header = False)


if __name__ == "__main__":
    argv = sys.argv[1:]
    inputfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
#     print(inputfile)
    main(inputfile, outputfile)