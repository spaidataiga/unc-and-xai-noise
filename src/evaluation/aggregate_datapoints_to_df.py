import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score
from scipy.stats import entropy
from datasets import load_from_disk
import sys
from ..utils.evaluation_utils import *
import warnings



warnings.filterwarnings("ignore") #### Keep getting a warning that makes the output log several GBs long


if __name__ == "__main__":
    datasets = [sys.argv[1]]
    stats = []
    pt = None
    
    with open('../../Resources/models.txt') as f:
        models = [line.strip() for line in f.readlines()]
        # models = ['opt']
        
    # with open('../../Resources/datasets.txt') as f:
    #     datasets = [line.strip() for line in f.readlines()]

    grads = ['GuidedBP', 'InputXGrad', 'IntegGrad', 'SmoothGrad']
    nks = ['token-unk', 'token-mask', 'charswap', 'synonym', 'butterfingers', 'wordswap', 'charinsert', 'l33t']
    lvls = ['05', '10', '25', '50', '70', '80', '90', '95'] 


    # First clean
    PATTERN = 'Clean'

                    
    for model in models:
        for dataset in datasets:
                        stats.append(collect_stats(model, dataset, PATTERN, temp_fix=True))


    PATTERN = 'random'

    for model in models:
        for dataset in datasets:
            for nk in nks: # kinds of noise
                for lvl in lvls: # levels of noise
                    stats.append(collect_stats(model, dataset, PATTERN, nk=nk, lvl=lvl, temp_fix=True))

    PATTERN = 'gradient'

    for model in models:
        for dataset in datasets:
            for nk in nks: # kinds of noise
                for lvl in lvls: # levels of noise
                    stats.append(collect_stats(model, dataset, PATTERN, nk=nk, lvl=lvl, temp_fix=True))
                    
    PATTERN = 'human'

    for model in models:
        for dataset in datasets:
            for pt in ['S', 'R']:
                for nk in nks: # kinds of noise
                    for lvl in lvls: # levels of noise
                        stats.append(collect_stats(model, dataset, PATTERN, nk=nk, lvl=lvl, pt=pt, temp_fix=True))

    
    PATTERN = 'human'
    pt = 'A'

    for model in models:
        for dataset in datasets:
            for nk in nks: # kinds of noise
                    stats.append(collect_stats(model, dataset, PATTERN, nk=nk, lvl=lvl, pt=pt, temp_fix=True))


    columns = ['pattern', 'pattern_type', 'model', 'data', 'perturbation', 'lvl', 'accuracy', 'softmax_prob', 'softmax_entropy', 'certainty_mean', 'certainty_variance', 'entropy', 'mutual_info'] + [k+"_og_corr" for k in grads] + [k+"_noise_corr" for k in grads] + [k+"_anno_corr" for k in grads] + [k+"_noise_MAP" for k in grads] + [k+"_all_anno_MAP" for k in grads]+ [k+"_majo_anno_MAP" for k in grads] # + [k+"_entropy" for k in grads] + [k+"_MAV" for k in grads] #+ [k+"_ZVD" for k in grads]

    try:
        df_output = pd.DataFrame(stats, columns= columns)
    except ValueError:
        df_output = pd.DataFrame(stats)
        print(f"More columns than expected.{len(columns)} columns expected, but {df_output.shape[1]} columns received. Saving without naming columns")
        print("Columns should be in the following order:")
        print(columns)

    df_output.to_csv(f"../../Results/{dataset}.csv")