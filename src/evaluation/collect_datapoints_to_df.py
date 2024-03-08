"""
Collect all results into a single DF to calculate correlations at a datapoint levels
"""

import os
from ast import literal_eval
import pandas as pd
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import entropy
from ..utils.evaluation_utils import collect_df
import sys

prefix = "/tmp/svm-erda/"

models = ['opt','bert', 'electra', 'roberta', 'gpt2-medium']
datasets = ['SST-2', 'SemEval', 'hatexplain']
lvls = ['05', '10', '25', '50', '70','80', '90', '95']
nks = ['token-unk', 'token-mask', 'charswap', 'synonym', 'butterfingers',  'charinsert', 'l33t'] #,'wordswap']
grads = ['GuidedBP', 'InputXGrad', 'IntegGrad', 'SmoothGrad']

if __name__ == "__main__":
    dfs = []
    for m in models:
        for d in datasets:
            dfs.append(collect_df(m, d, 'Clean'))
            for nk in nks:
                for lvl in lvls:
                    for pattern in ['human', 'gradient', 'random']:
                        if pattern == 'human':
                            pt = 'S'
                        else:
                            pt= None
                        dfs.append(collect_df(m, d, pattern, nk=nk, lvl=lvl, pt=pt))
                        
                        
    out = pd.concat(dfs)
    out.to_csv(f"All_datapoints.tsv", sep= "\t")
                        