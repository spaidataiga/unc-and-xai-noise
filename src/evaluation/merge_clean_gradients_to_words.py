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

grads = ['GuidedBP', 'InputXGrad', 'IntegGrad', 'SmoothGrad']
# models = [sys.argv[0]]
# datasets = [sys.argv[1]]

#######################################################################################################################

if __name__ == "__main__":
    # 
    for model in models:
        for dataset in datasets:
            if dataset == 'SST-2':
                text_label = "sentence"
                test_data = dataset
            else:
                text_label = "text"
                test_data = dataset

            ## Get Data
            for c in range(2):

                for k in grads:
                    bert_data = pd.read_csv(f"{prefix}Saliency/RAW/{model}/Classes/{dataset}/{c}/{k}.tsv", sep="\t") # dataframe # pd.read_csv(f"Saliency/Noise0/RAW/{model}/{dataset}-{k}.tsv", sep="\t") # dataframe
                    # orig_data = load_from_disk(f"./Data/Clean/{test_data}")['test'][text_label] # text form
                    
                    test = pd.DataFrame(load_from_disk(f"../../Data/Clean/{test_data}")['test']['text'], columns=['text'])
                    test.index = load_from_disk(f"./Data/Clean/{test_data}")['test']['index']
                    orig_data = test.loc[bert_data['id'].tolist()]['text']


                    word2attributions, pred_token_dic, new_data_record = process_bert(bert_data, orig_data, model)

                    directory = f"{prefix}Saliency/Clean/{model}/Classes/{dataset}/{c}/"
                    if not os.path.exists(directory):
                    # If it doesn't exist, create it
                        os.makedirs(directory)
                        
                    visualize_text_pred(new_data_record, bert_data.id.to_list(), bert_data.pred_entropy.to_list(), f"{directory}{k}")
