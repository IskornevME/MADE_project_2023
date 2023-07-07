"""
This module takes:
    - file with docs + gen text (for example 'data_for_metrics_rugpt_1000.json');
    - file with scores (got him with command from orgs);
    - output file for metrics.
Usage: python count_metric_script.py -i data_for_metrics_rugpt_1000.json -s scores_m3_rugpt.txt -o metrics_rugpt.json
"""
import sys, getopt
import pandas as pd
import numpy as np
import time
import warnings
import json
import seaborn as sns

from rank_bm25 import BM25Okapi
from scipy import stats, special
from matplotlib import pyplot as plt
from collections import defaultdict
from tqdm import tqdm


FAKE_DOC_LABEL = [-1]
RELEVANT_DOC_LABELS = [2, 3]


def get_scores(scoresfile, df):
    df_tmp = df.copy()
    with open(scoresfile, "r") as s:
        scores = s.readlines()
    for i in range(len(scores)):
        tmp = scores[i]
        scores[i] = float(tmp.split('\\')[0])
    df_tmp["scores"] = scores
    return df_tmp


def count_fdaro_arl(df):
    df_tmp = df.copy()
    FDARO_v1 = []
    FDARO_v2 = []
    ARL = []
    for row in tqdm(df_tmp.iterrows()):
        scores = row[1]["scores"]
        labels = row[1]["label"]
        
        ranking_list = sorted([item for item in zip(scores, labels)], key=lambda x: x[0], reverse=True)

        is_first = False
        for item in ranking_list:
            if item[1] in RELEVANT_DOC_LABELS:
                break
            elif item[1] in FAKE_DOC_LABEL:
                is_first = True
                break

        if is_first:
            FDARO_v1.append(1.)
        else:
            FDARO_v1.append(0.)
            
        scores_relevant = -1e9
        scores_fake = -1e9
        for ind in range(len(labels)):
            if labels[ind] in RELEVANT_DOC_LABELS:
                scores_relevant = scores[ind]
            elif labels[ind] in FAKE_DOC_LABEL and scores_fake == -1e9:
                scores_fake = scores[ind]

        upper_or_not = (scores_fake - scores_relevant) > 1e-12
        
        FDARO_v2.append(int(upper_or_not))
        
        is_fake = False
        for ind, item in enumerate(ranking_list):
            if item[1] in FAKE_DOC_LABEL:
                ARL.append((ind + 1) / len(ranking_list))
                is_fake = True

        if not is_fake:
            ARL.append(1.)

    df_tmp["FDARO_v1"] = FDARO_v1
    df_tmp["FDARO_v2"] = FDARO_v2
    df_tmp["AverageRelLoc"] = ARL
    return df_tmp


def main(inputfile, scoresfile, outputfile):
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
    df_processed_exp = get_scores(scoresfile, df_processed_exp)
    df_processed_2 = df_processed_exp.groupby('query').agg(lambda x: list(x)).reset_index()
    
    df_processed_m = df_processed_2.copy()
    df_processed_m = count_fdaro_arl(df_processed_m)
    
    mean_fdaro_v1 = df_processed_m["FDARO_v1"].mean()
    mean_fdaro_v2 = df_processed_m["FDARO_v2"].mean()
    mean_arl = df_processed_m["AverageRelLoc"].mean()
    
    data_json = {"FDARO_v1": mean_fdaro_v1, "FDARO_v2": mean_fdaro_v2, "AverageRelLoc": mean_arl}
    
    print(f"FDARO_v1 = {np.round(mean_fdaro_v1, 4)}")
    print(f"FDARO_v2 = {np.round(mean_fdaro_v2, 4)}")
    print(f"AverageRelLoc = {np.round(mean_arl, 4)}")
    
    with open(outputfile, "w") as outfile:
        json.dump(data_json, outfile)
    

if __name__ == "__main__":
    argv = sys.argv[1:]
    inputfile = ''
    scoresfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv,"hi:s:o:",["ifile=","sfile=", "ofile="])
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -s <scoresfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-s", "--sfile"):
            scoresfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
#     print ('Input file is ', inputfile)
#     print ('Scores file is ', scoresfile)
#     print ('Output file is ', outputfile)
    main(inputfile, scoresfile, outputfile)