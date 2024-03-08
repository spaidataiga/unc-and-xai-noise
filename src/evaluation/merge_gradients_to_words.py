import pandas as pd
from datasets import load_from_disk
from ..utils.gradient_utils import *
from ..utils.model_utils import *
import os
import sys

prefix = "/tmp/svm-erda/" #### Save to directly to cloud ERDA

noise_folder = "OPTNoise/" #"Noise/"

with open('../../Resources/models.txt') as f:
    models = [line.strip() for line in f.readlines()]
    
with open('../../Resources/datasets.txt') as f:
    datasets = [line.strip() for line in f.readlines()]

pts = ['S', 'R']
nks = ['token-unk','token-mask', 'charswap', 'synonym', 'butterfingers', 'wordswap', 'charinsert', 'l33t']
lvls = ['05', '10', '25', '50', '70', '80', '90', '95']
grads = ['GuidedBP', 'InputXGrad', 'IntegGrad', 'SmoothGrad']

#######################################################################################################################

if __name__ == "__main__":

    PATTERN = 'human'

    for model in models:
        for dataset in datasets:
            for pt in pts:
                for nk in nks: # kinds of noise
                    for lvl in lvls: # levels of noise
                            if "token" in nk:
                                text_label = f"{PATTERN}-{pt}_{nk.split('-')[0]}_{lvl}"
                                token_type = nk.split('-')[1]
                            else:
                                text_label = f"{PATTERN}-{pt}_{nk}_{lvl}" # human-R_token_{prop*100:2.0f}'
                                token_type = None
                            test_data = dataset

                            ## Get Data

                            for k in grads:
                                try:
                                    bert_data = pd.read_csv(f"{prefix}Saliency/RAW/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}/{k}.tsv", sep="\t", quoting=3) # dataframe
                                    # orig_data = load_from_disk(f"./Data/Noise/{test_data}")[text_label] # text form
                                    
                                    orig_data_unsorted = pd.DataFrame(load_from_disk(f"./Data/{noise_folder}{test_data}")[text_label], columns=['text'])
                                    orig_data_unsorted.index = load_from_disk(f"./Data/{noise_folder}{test_data}")['index']
                                    orig_data = orig_data_unsorted.loc[bert_data['id'].tolist()]['text']

                                    word2attributions, pred_token_dic, new_data_record = process_bert(bert_data, orig_data, model, token_type)

                                    directory = f"{prefix}Saliency/Clean/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}/{k}/"
                                    if not os.path.exists(directory):
                                    # If it doesn't exist, create it
                                        os.makedirs(directory)
                                        
                                    visualize_text_pred(new_data_record, bert_data.id.to_list(), bert_data.pred_entropy.to_list(), f"{directory}{k}")
                                except (FileNotFoundError, ValueError) as err:
                                    print(f"NOT EXISTANT: {prefix}Saliency/RAW/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}/{k}.tsv")
                                    print(err)

    PATTERN = 'human'

    for model in models:
        for dataset in datasets:
            pt = 'A'
            for nk in nks: # kinds of noise
                
                if "token" in nk:
                    text_label = f"{PATTERN}-{pt}_{nk.split('-')[0]}"
                    token_type = nk.split('-')[1]
                else:
                    text_label = f"{PATTERN}-{pt}_{nk}" # human-R_token_{prop*100:2.0f}'
                    token_type = None
                            
                # text_label = f"{PATTERN}-{pt}_{nk.split('-')[0]}"
                test_data = dataset

                ## Get Data

                for k in grads:
                    try:
                        bert_data = pd.read_csv(f"{prefix}Saliency/RAW/{model}/{PATTERN}/{pt}/{nk}/{dataset}/{k}.tsv", sep="\t", quoting=3) # dataframe
                        # orig_data = load_from_disk(f"./Data/{noise_folder}{test_data}")[text_label] # text form
                        
                        orig_data_unsorted = pd.DataFrame(load_from_disk(f"./Data/{noise_folder}{test_data}")[text_label], columns=['text'])
                        orig_data_unsorted.index = load_from_disk(f"./Data/{noise_folder}{test_data}")['index']
                        orig_data = orig_data_unsorted.loc[bert_data['id'].tolist()]['text']
                                
                        word2attributions, pred_token_dic, new_data_record = process_bert(bert_data, orig_data, model, token_type)


                        directory = f"{prefix}Saliency/Clean/{model}/{PATTERN}/{pt}/{nk}/{dataset}/{k}/"
                        if not os.path.exists(directory):
                        # If it doesn't exist, create it
                            os.makedirs(directory)
                        if os.path.isfile(f"{directory}{k}.tsv"):
                                print(f"DATA EXISTS ALREADY FOR {directory}{k}")
                                continue
                        else:
                            visualize_text_pred(new_data_record, bert_data.id.to_list(), bert_data.pred_entropy.to_list(), f"{directory}{k}")
                    except (FileNotFoundError, ValueError) as err:
                        print(f"NOT EXISTANT: {prefix}Saliency/RAW/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}/{k}.tsv")
                        print(err)                
    PATTERN = 'random'

    for model in models:
        for dataset in datasets:
            for nk in nks: # kinds of noise
                for lvl in lvls: # levels of noise
            
                    if "token" in nk:
                        text_label = f"{PATTERN}_{nk.split('-')[0]}_{lvl}"
                        token_type = nk.split('-')[1]
                    else:
                        text_label = f"{PATTERN}_{nk}_{lvl}" # human-R_token_{prop*100:2.0f}'
                        token_type = None
                    test_data = dataset

                    ## Get Data

                    for k in ['GuidedBP', 'InputXGrad', 'IntegGrad', 'SmoothGrad']:
                        try:
                            bert_data = pd.read_csv(f"{prefix}Saliency/RAW/{model}/{PATTERN}/{nk}/{lvl}/{dataset}/{k}.tsv", sep="\t", quoting=3) # dataframe
                            # orig_data = load_from_disk(f"./Data/{noise_folder}{test_data}")[text_label] # text form
                            orig_data_unsorted = pd.DataFrame(load_from_disk(f"./Data/{noise_folder}{test_data}")[text_label], columns=['text'])
                            orig_data_unsorted.index = load_from_disk(f"./Data/{noise_folder}{test_data}")['index']
                            orig_data = orig_data_unsorted.loc[bert_data['id'].tolist()]['text']
                            
                            word2attributions, pred_token_dic, new_data_record = process_bert(bert_data, orig_data, model, token_type)

                            directory = f"{prefix}Saliency/Clean/{model}/{PATTERN}/{nk}/{lvl}/{dataset}/{k}/"
                            if not os.path.exists(directory):
                            # If it doesn't exist, create it
                                os.makedirs(directory)
                            if os.path.isfile(f"{directory}{k}.tsv"):
                                print(f"DATA EXISTS ALREADY FOR {directory}{k}")
                                continue
                            else:
                                visualize_text_pred(new_data_record, bert_data.id.to_list(), bert_data.pred_entropy.to_list(), f"{directory}{k}")
                        except (FileNotFoundError, ValueError) as err:
                            print(f"NOT EXISTANT: {prefix}Saliency/RAW/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}/{k}.tsv")
                            print(err)
    PATTERN = 'gradient'

    for model in models:
        for dataset in datasets:
            for nk in nks: # kinds of noise
                for lvl in lvls: # levels of noise
        
                    if "token" in nk:
                        text_label = f"{PATTERN}-{model}_{nk.split('-')[0]}_{lvl}"
                        token_type = nk.split('-')[1]
                    else:
                        text_label = f"{PATTERN}-{model}_{nk}_{lvl}" 
                        token_type = None
                    test_data = dataset

                    ## Get Data

                    for k in ['GuidedBP', 'InputXGrad', 'IntegGrad', 'SmoothGrad']:
                        try:
                            bert_data = pd.read_csv(f"{prefix}Saliency/RAW/{model}/{PATTERN}/{nk}/{lvl}/{dataset}/{k}.tsv", sep="\t", quoting=3) # dataframe
                            # orig_data = load_from_disk(f"./Data/{noise_folder}{test_data}")[text_label] # text form
                            
                            orig_data_unsorted = pd.DataFrame(load_from_disk(f"./Data/{noise_folder}{test_data}")[text_label], columns=['text'])
                            orig_data_unsorted.index = load_from_disk(f"./Data/{noise_folder}{test_data}")['index']
                            orig_data = orig_data_unsorted.loc[bert_data['id'].tolist()]['text']
                        
                            word2attributions, pred_token_dic, new_data_record = process_bert(bert_data, orig_data, model, token_type)

                            directory = f"{prefix}Saliency/Clean/{model}/{PATTERN}/{nk}/{lvl}/{dataset}/{k}/"
                            if not os.path.exists(directory):
                            # If it doesn't exist, create it
                                os.makedirs(directory)
                            if os.path.isfile(f"{directory}{k}.tsv"):
                                print(f"DATA EXISTS ALREADY FOR {directory}{k}")
                                continue
                            else:   
                                visualize_text_pred(new_data_record, bert_data.id.to_list(), bert_data.pred_entropy.to_list(), f"{directory}{k}")
                        except (FileNotFoundError, ValueError) as err:
                            print(f"NOT EXISTANT: {prefix}Saliency/RAW/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}/{k}.tsv")
                            print(err)

