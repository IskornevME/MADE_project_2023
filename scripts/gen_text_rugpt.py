"""
This module takes file with clear docs (for example 'test_sample_1000.json') to get queries
and than generate text. After we concate it.
Usage: python gen_text_rugpt.py -i test_sample_1000.json -m train -o experiment_0_rugpt_data_for_metrics_3_samples.json -n rugpt
"""
import torch
import json
import random
import time
import configparser
import numpy as np
import pandas as pd
import sys, getopt

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from tqdm.notebook import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict
from tqdm import tqdm


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            print(input_ids)
            return True
        return False


def union_files(inputfile, file_with_gen_text_name, outputfile):
    ########################################
    GEN_DOCS_NAMES = [file_with_gen_text_name]
    SAMPLE_NAME = inputfile
    OUTPUT_NAMES = [outputfile]
    ########################################

    assert len(GEN_DOCS_NAMES) == len(OUTPUT_NAMES)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)


    with open(SAMPLE_NAME, "r", encoding='utf-8') as infile:
        sample = json.load(infile)
        
    sample = sample[:2]

    for j, GEN_DOCS_NAME in enumerate(GEN_DOCS_NAMES):
        with open(GEN_DOCS_NAME, "r", encoding='utf-8') as infile:
            gen_docs = json.load(infile)

        assert len(sample) == len(gen_docs)

        data_for_metrics = []
        for i in range(len(sample)):
            elem = {}
            elem['query'] = sample[i]['query']
            elem['passage_text'] = sample[i]['body'].copy()
            elem['is_selected'] = sample[i]['label'].copy()

            assert gen_docs[i]['query'] == sample[i]['query']

            elem['passage_text'].append(gen_docs[i]['generated_doc'])
            elem['is_selected'].append(-1)

            data_for_metrics.append(elem)

        assert len(data_for_metrics) == len(sample)

        with open(OUTPUT_NAMES[j], "w", encoding='utf-8') as outfile:
            json.dump(data_for_metrics, outfile, indent = 4, ensure_ascii=False, cls=NpEncoder)

        print(f'File {GEN_DOCS_NAME} processed!')
    
    
def get_model_params(config, model_name):
    model_params = {}
    model_params["max_length"] = int(config.get(model_name,'max_length'))
    model_params["num_beams"] = int(config.get(model_name,'num_beams'))
    model_params["repetition_penalty"] = float(config.get(model_name,'repetition_penalty'))
    model_params["temperature"] = float(config.get(model_name,'temperature'))
    model_params["top_p"] = float(config.get(model_name,'top_p'))
    model_params["top_k"] = int(config.get(model_name,'top_k'))
    model_params["no_repeat_ngram_size"] = int(config.get(model_name,'no_repeat_ngram_size'))
    return model_params


def main(inputfile, outputfile, pretrain_flg, model_name):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = configparser.ConfigParser()
    config.read('configfile')
    with open(inputfile, 'r') as f:
        data = json.load(f)
    queries = []
    for elem in data:
        queries.append(elem['query'])
#     model_params = {}
    if pretrain_flg == 1:
        PRETRAIN_DIR = config.get('pmp','pretrain_model_path')
        tokenizer = GPT2Tokenizer.from_pretrained(f'{PRETRAIN_DIR}tokenizer')
        model = GPT2LMHeadModel.from_pretrained(f'{PRETRAIN_DIR}model_with_summary').to(DEVICE)
        model_params = get_model_params(config, "rugpt")
    else:
        if model_name == "rugpt":
            tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
            model = GPT2LMHeadModel.from_pretrained("ai-forever/rugpt3small_based_on_gpt2").to(DEVICE)
            model_params = get_model_params(config, "rugpt")
        else:
            tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/mGPT")
            model = GPT2LMHeadModel.from_pretrained("ai-forever/mGPT").to(DEVICE)
            model_params = get_model_params(config, "mgpt")
#     print(PRETRAIN_DIR)
    SPECIAL_TOKENS = {'bos_token':'<bos>','eos_token' :'<eos>', 'pad_token':'<pad>', 'sep_token': '<sep>'}
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    stop_criteria = KeywordsStoppingCriteria(tokenizer.encode(tokenizer.eos_token, return_tensors="pt").to(DEVICE))
    
    generated = []
    cnt = 0
    result_file = []
    with torch.no_grad():
        for query in tqdm(queries[:2]):
#         for query in tqdm(queries):
            out = model.generate(tokenizer.encode(query + " ", return_tensors="pt").to(DEVICE),
                              do_sample=True,
                              num_beams=model_params["num_beams"],
                              repetition_penalty=model_params["repetition_penalty"],
                              temperature=model_params["temperature"],
                              top_p=model_params["top_p"],
                              max_length = model_params["max_length"],
                              top_k=model_params["top_k"],
                              no_repeat_ngram_size=model_params["no_repeat_ngram_size"],
                              stopping_criteria=StoppingCriteriaList([stop_criteria]),
                              eos_token_id=tokenizer.eos_token_id,
                              bos_token_id=tokenizer.bos_token_id,
                              ).to(DEVICE)
            gen_doc = tokenizer.batch_decode(out, skip_special_tokens=False)[0][len(query) + 1:]
            cnt += 1

            result_file.append({'query': query, 'generated_doc': gen_doc})
#             print(query)
#             print(gen_doc)
            
    file_with_gen_text_name = "gen_text.json"
    with open(file_with_gen_text_name, 'w') as f:
        json.dump(result_file, f, indent=4, ensure_ascii=False)
    union_files(inputfile, file_with_gen_text_name, outputfile)
    

if __name__ == "__main__":
    argv = sys.argv[1:]
    inputfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv,"hi:m:o:n:",["ifile=", "mode=", "ofile=", "name="])
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-m", "--mode"):
            pretrain = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-n", "--name"):
            model_name = arg
    print(model_name)
    print(inputfile)
    pretrain_flg = 0
    if pretrain == "pretrain":
        pretrain_flg = 1
    print(pretrain_flg)
    main(inputfile, outputfile, pretrain_flg, model_name)