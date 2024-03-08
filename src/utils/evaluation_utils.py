import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score
from scipy.stats import entropy
from datasets import load_from_disk
from ast import literal_eval
import os


################################# DEFINE GLOBAL VARIABLES ######################################

prefix = "/tmp/svm-erda/"
NoiseFolder =  "Noise" #"OPTNoise" #Noise
grads = ['GuidedBP', 'InputXGrad', 'IntegGrad', 'SmoothGrad']

##############################################################################

def get_correlation(new, old, temp_fix=False, extra_data=None):
    
    if np.all(old==1):
        print("IMPOSSIBLE TO ANNOTATE:", extra_data)
        return np.nan
    
    if temp_fix == True:
        if len(old) > len(new): ## If tokenizer wrong size, do this for now
            print("FIXED: ", extra_data)
            old = old[:len(new)]
    try:
        return pearsonr(new, old)[0]
    except ValueError:
        print("Gave 0:", extra_data)
        return np.nan

def get_map(new, old, temp_fix=False, extra_data=None):
    global high_prec
    
    if not any(old):
        print(extra_data)
        #old = 
        return np.nan
    
    if temp_fix == True:
        if len(old) > len(new): ## If tokenizer wrong size, do this for now
            print("FIXED: ", extra_data)
            old = old[:len(new)]
        if len(new) > len(old): ### WHAT CAUSES THIS?
            print("LONGER NEW TEST: ", extra_data)
            new = new[:len(old)]
            
    return average_precision_score(new,old,pos_label=1)

def convert_to_list(val):
    return [float(x) for x in val.split(' ')]

def get_entropy_complexity(x):
    a = np.array(np.abs(x), dtype=np.float64) / np.sum(np.abs(x))
    return entropy(pk=a)

def get_zero_vd(x):
    ref = np.zeros(x.shape)
    return np.linalg.norm(x - ref)

def collect_stats(model, dataset, PATTERN, nk=None, lvl=None, pt=None, temp_fix=False):
    global grads
    if PATTERN == 'Clean':
        filename = f"{prefix}Uncertainty/{model}/{PATTERN.upper()}/{dataset}_uncertainty.tsv"
    elif PATTERN == 'human':
        if pt == 'A':
            filename = f"{prefix}Uncertainty/{model}/{PATTERN}/{pt}/{nk}/{dataset}_uncertainty.tsv"
        else:
            filename = f"{prefix}Uncertainty/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}_uncertainty.tsv"
    else: ## All else
        filename = f"{prefix}Uncertainty/{model}/{PATTERN}/{nk}/{lvl}/{dataset}_uncertainty.tsv"
    
    if dataset == 'hatexplain' or dataset == 'esnli':
        n_classes = 3
    else:
        n_classes = 2
        
    if model == "opt":
        NoiseFolder =  "OPTNoise"
    else:
        NoiseFolder =  "Noise" #"OPTNoise" #Noise
        
    high_prec = pd.read_csv("High_Precision_CLEAN.tsv", sep="\t", index_col=0,quoting=3)

    try:
        df = pd.read_csv(filename, index_col=0,quoting=3)

        # if (dataset == 'hatexplain') and( temp_fix == True): ############ CATCH
        #     print("REMOVING DATAPOINTS 304,407,932 DUE TO CORRUPTION")
        #     df.drop(index=[304,407,932], inplace = True)  
            
        certainty_mean = df[[f'mean_{n}' for n in range(n_classes)]].max(axis=1).mean()
        entropy_val = df[['entropy']].max(axis=1).mean()
        mutual_info = df[['mutual_info']].max(axis=1).mean()

        COI = 'variance_'+ df[[f'mean_{n}' for n in range(n_classes)]].idxmax(axis=1).map(lambda x: x[-1]).values

        variances = []
        for i in range(df.shape[0]):
            variances.append(df[COI[i]].iloc[i])
        certainty_variance = np.mean(variances)
        
    except (FileNotFoundError, ValueError) as error:
        print("MISSING: ", filename)
        print(error)
        certainty_mean = np.nan
        certainty_variance = np.nan
        entropy_val = np.nan
        mutual_info = np.nan

    og_sims = []
    noise_sims = []
    anno_sims = []
    noise_map = []
    all_anno_map = []
    majo_anno_map = []
    complexities = []
    mavs = []
    # zvds = []

    if dataset == 'SST-2':                
        anno_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"annotations"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="anno").map(lambda x: np.array([float(y) for y in x.split()]))
    elif dataset == 'esnli':
        out = pd.DataFrame([load_from_disk(f"./Data/Noise/esnli")[f"annotation_1"],load_from_disk(f"./Data/Noise/esnli")[f"annotation_2"]]).T
        out.index = load_from_disk(f"./Data/Noise/esnli")["index"]
        out['anno'] = out.apply(lambda x: np.array(x[0] + [0.0] + x[1]), axis=1)
        anno_pattern = out['anno']
        # anno_pattern = anno_pattern.map(lambda x: np.array([float(y) for y in x.split()]))
        #     test = pd.DataFrame(zip(load_from_disk(f"./Data/Clean/{test_data}")['test']['text_1'], load_from_disk(f"./Data/Clean/{test_data}")['test']['text_1']), columns=text_label)
        # test['text'] = test['text_1'] + ' | ' + test['text_2']
        # test.index = load_from_disk(f"./Data/Clean/{test_data}")['test']['index']
        # orig_data = test.loc[bert_data['id'].tolist()]['text']
    else:
        anno_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"annotations"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="anno").map(lambda x: np.array([float(y) for y in x]))
            
    label_pattern =  pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"label"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="label").map(lambda x: x*2 - 1)

    anno_pattern *= label_pattern
    anno_pattern.name = 'anno'
    
    if PATTERN == 'human':
        if pt == 'A':
            perturb_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"{PATTERN}-{pt}_PATTERN"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="perturbation")
        else:
            perturb_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"{PATTERN}-{pt}_PATTERN_{lvl}"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="perturbation")
    elif PATTERN == 'gradient':
        perturb_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"{PATTERN}-{model}_PATTERN_{lvl}"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="perturbation")
    elif PATTERN == 'random': # random
        perturb_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"{PATTERN}_PATTERN_{lvl}"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="perturbation")
    
    if PATTERN != "Clean":
        perturb_pattern = perturb_pattern.map(lambda x: [int(literal_eval(y)) for y in x.split()])

        
    all_annos_pattern= anno_pattern.map(lambda x: np.where(x>0, 1, 0))
    majority_annos_pattern = anno_pattern.map(lambda x: np.where(x>0.5, 1, 0))
    all_annos_pattern.name = 'anno'
    majority_annos_pattern.name = 'anno'

    accuracy_mean = None # To test if a value had already been written
    for k in grads:
        try:
            og_filename = f"{prefix}Saliency/Updated/{model}/{dataset}-{k}.tsv"
            if not os.path.exists(og_filename):
                og_filename = og_filename.replace("Updated", "Clean")
            df_og = pd.read_csv(og_filename, index_col=0, sep = '\t',quoting=3)
            
        
            if PATTERN != 'Clean':
                if PATTERN == 'human':
                    if pt == 'A':
                        filename = f"{prefix}Saliency/Updated/{model}/{PATTERN}/{pt}/{nk}/{dataset}/{k}/{k}.tsv"
                        if not os.path.exists(filename):
                            filename = filename.replace("Updated", "Clean")
                        df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
                    else:
                        filename = f"{prefix}Saliency/Updated/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}/{k}/{k}.tsv"
                        if not os.path.exists(filename):
                            filename = filename.replace("Updated", "Clean")
                        df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
                else:
                    filename = f"{prefix}Saliency/Updated/{model}/{PATTERN}/{nk}/{lvl}/{dataset}/{k}/{k}.tsv"
                    if not os.path.exists(filename):
                        filename = filename.replace("Updated", "Clean")
                    df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
                    
            else:
                filename = f"{prefix}Saliency/Updated/{model}/{dataset}-{k}.tsv"
                if not os.path.exists(filename):
                    filename = filename.replace("Updated", "Clean")
                df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
            
            accuracy_mean = 1 - (df['true_class'] - df['pred_class']).abs().mean()
            pred_prob = df['pred_prob'].mean()
            pred_entropy = df['pred_entropy'].mean()
            
            # if dataset == 'hatexplain' and temp_fix:
            #     df.drop(index=[304,407,932], inplace= True)
            #     df_og.drop(index=[304,407,932], inplace= True)
            
            df['attribution_scores'] = df['attribution_scores'].map(convert_to_list)
            df_og['attribution_scores'] = df_og['attribution_scores'].map(convert_to_list)
            
            ## Get correlation to original saliency map
            if PATTERN == 'Clean':
                og_sims.append(1.0) # Perfect correlation
            else:
                og_corrs = df[['attribution_scores']].merge(df_og[['attribution_scores']], on ='id', how = 'inner').dropna().apply(lambda x: get_correlation(x.attribution_scores_x, x.attribution_scores_y, temp_fix=True, extra_data=[dataset,PATTERN, nk, k,lvl, x.index]), axis=1)
                og_sims.append(np.mean(og_corrs))
                
            if PATTERN == 'Clean':
                ## Get correlation to perturbation map
                noise_sims.append(np.nan) ## No perturbation in Clean data
                noise_map.append(np.nan)
            else:
                ## Get correlation to perturbation map
                noise_corrs = df[['attribution_scores']].merge(perturb_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_correlation(x.attribution_scores, x.perturbation, temp_fix=True, extra_data=[dataset,PATTERN, nk, k, lvl, x.index]), axis=1)
                noise_sims.append(np.mean(noise_corrs))
                noise_corrs = df[['attribution_scores']].merge(perturb_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_map( x.perturbation, x.attribution_scores, temp_fix=True, extra_data=[dataset,PATTERN, nk,k, lvl, x.index]), axis=1)
                noise_map.append(np.mean(noise_corrs))
                
            if "_neutral" in dataset:
                anno_sims.append(np.nan)
                all_anno_map.append(np.nan)
                majo_anno_map.append(np.nan)

            else:
                ## Get correlation to humman annotation
                anno_corrs = df[['attribution_scores']].merge(anno_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_correlation(x.attribution_scores, x.anno, temp_fix=True, extra_data=[dataset,PATTERN, nk,k, lvl, x.index]), axis=1) #temp_fix=False, extra_data=None):
                anno_sims.append(np.mean(anno_corrs))
                
                ## Get correlation to humman annotation
                all_anno_corrs = df[['attribution_scores']].merge(all_annos_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_map( x.anno, x.attribution_scores, temp_fix=True, extra_data=[dataset,PATTERN, nk,k, lvl, x.index]), axis=1)
                all_anno_map.append(np.mean(all_anno_corrs))
                
                majo_anno_corrs = df[['attribution_scores']].merge(majority_annos_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_map( x.anno, x.attribution_scores, temp_fix=True, extra_data=[dataset,PATTERN, nk,k, lvl, x.index]), axis=1)
                majo_anno_map.append(np.mean(majo_anno_corrs))
            
            # complexity = df['attribution_scores'].map(get_entropy_complexity)
            # complexities.append(np.mean(complexity))
            
            # mav = df['attribution_scores'].map(lambda x: np.mean(np.abs(x)))
            # mavs.append(np.mean(mav))
            
            # zvd = df['attribution_scores'].map(lambda x: get_zero_vd(x))
            # zvds.append(np.mean(zvd))
            
        except (FileNotFoundError, ValueError) as error:
            complexities.append(np.nan)
            majo_anno_map.append(np.nan)
            all_anno_map.append(np.nan)
            noise_map.append(np.nan)
            noise_sims.append(np.nan)
            og_sims.append(np.nan)
            if not accuracy_mean:
                accuracy_mean = np.nan
                pred_prob = np.nan
                pred_entropy = np.nan
            print(f"Missing filename: {filename}")
            print(error)
            
    out = [PATTERN, pt, model, dataset, nk, lvl, accuracy_mean, pred_prob, pred_entropy, certainty_mean, certainty_variance, entropy_val, mutual_info] + og_sims + noise_sims + anno_sims + noise_map + all_anno_map + majo_anno_map # + complexities + mavs #+ zvds
    if len(out) != 37:
        print("PATTERN: ", PATTERN)
        print("pt: ", pt)
        print("model: ", model)
        print("dataset: ", dataset)
        print("nk: ", nk)
        print("lvl: ", lvl)
        print("accuracy_mean: ", accuracy_mean)
        print("pred_prob: ", pred_prob)
        print("pred_entropy: ", pred_entropy)
        print("certainty_mean: ", certainty_mean)
        print("certainty_variance: ", certainty_variance)
        print("entropy_val: ", entropy_val)
        print("mutual_info: ", mutual_info)
        print("og_sims: ", og_sims)
        print("noise_sims: ", noise_sims)
        print("anno_sims: ", anno_sims)
        print("noise_map: ", noise_map)
        print("all_anno_map: ", all_anno_map)
        print("majo_anno_map: ", majo_anno_map)
        # print("complexities: ", complexities)      
        # print("mavs: ", mavs)
        #print("zvds: ", mavs)            
            
    return out


def collect_df(model, dataset, PATTERN, nk=None, lvl=None, pt=None):
    if PATTERN == 'Clean':
        filename = f"{prefix}Uncertainty/{model}/{PATTERN.upper()}/{dataset}_uncertainty.tsv"
    elif PATTERN == 'human':
        if pt == 'A':
            filename = f"{prefix}Uncertainty/{model}/{PATTERN}/{pt}/{nk}/{dataset}_uncertainty.tsv"
        else:
            filename = f"{prefix}Uncertainty/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}_uncertainty.tsv"
    else: ## All else
        filename = f"{prefix}Uncertainty/{model}/{PATTERN}/{nk}/{lvl}/{dataset}_uncertainty.tsv"

    try:
        df_unc = pd.read_csv(filename, index_col=0)

        # if (dataset == 'hatexplain') and( temp_fix == True): ############ CATCH
        #     print("REMOVING DATAPOINTS 304,407,932 DUE TO CORRUPTION")
        #     df.drop(index=[304,407,932], inplace = True)  
        
    except (FileNotFoundError, ValueError) as error:
        print(f"COULD NOT FIND UNCERTAINTY DATA FOR {filename}")
        return pd.DataFrame([])

    if dataset == 'SST-2':                
        anno_pattern = pd.Series(load_from_disk(f"./Data/Noise/{dataset}")[f"annotations"], index=load_from_disk(f"./Data/Noise/{dataset}")["index"], name="anno").map(lambda x: np.array([float(y) for y in x.split()]))
    elif dataset == 'esnli':
        out = pd.DataFrame([load_from_disk(f"./Data/Noise/esnli")[f"annotation_1"],load_from_disk(f"./Data/Noise/esnli")[f"annotation_2"]]).T
        out.index = load_from_disk(f"./Data/Noise/esnli")["index"]
        out['anno'] = out.apply(lambda x: np.array(x[0] + [0.0] + x[1]), axis=1)
        anno_pattern = out['anno']
    else:
        anno_pattern = pd.Series(load_from_disk(f"./Data/Noise/{dataset}")[f"annotations"], index=load_from_disk(f"./Data/Noise/{dataset}")["index"], name="anno").map(lambda x: np.array([float(y) for y in x]))
            
    label_pattern =  pd.Series(load_from_disk(f"./Data/Noise/{dataset}")[f"label"], index=load_from_disk(f"./Data/Noise/{dataset}")["index"], name="label").map(lambda x: x*2 - 1)

    anno_pattern *= label_pattern
    anno_pattern.name = 'anno'
        
        
    all_annos_pattern= anno_pattern.map(lambda x: np.where(x>0, 1, 0))
    majority_annos_pattern = anno_pattern.map(lambda x: np.where(x>0.5, 1, 0))
    all_annos_pattern.name = 'anno'
    majority_annos_pattern.name = 'anno'

    grad_dfs = []


    for k in grads:
        if PATTERN != 'Clean':
            if PATTERN == 'human':
                filename = f"{prefix}Saliency/Updated/{model}/{PATTERN}/S/{nk}/{lvl}/{dataset}/{k}/{k}.tsv"
                if not os.path.exists(filename):
                    filename = filename.replace("Updated", "Clean")
                df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
            else:
                filename = f"{prefix}Saliency/Updated/{model}/{PATTERN}/{nk}/{lvl}/{dataset}/{k}/{k}.tsv"
                if not os.path.exists(filename):
                    filename = filename.replace("Updated", "Clean")
                df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
                
        else:
            filename = f"{prefix}Saliency/Updated/{model}/{dataset}-{k}.tsv"
            if not os.path.exists(filename):
                filename = filename.replace("Updated", "Clean")
            df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
        
    
        df['attribution_scores'] = df['attribution_scores'].map(convert_to_list)
        # df['COM'] =  df['attribution_scores'].map(get_entropy_complexity)
        # df['ZVD'] =  df['attribution_scores'].map(get_zero_vd)
        with_annos = df.merge(all_annos_pattern, left_index=True, right_index=True)
        
        out = df_unc.merge(with_annos, right_index=True, left_index=True)[['pred_class', 'true_class', 'entropy', 'pred_entropy', 'attribution_scores', 'anno']] #'COM', 'ZVD', 'anno']]
        out['lvl'] = lvl
        out['model'] = model
        out['dataset'] = dataset
        out['nk'] = nk
        out['pattern'] = PATTERN
        out['saliency_map'] = k
        
        
            
        grad_dfs.append(out)
            
    return pd.concat(grad_dfs)

# def collect_df(model, dataset, PATTERN, nk=None, lvl=None, pt=None, temp_fix=False):
#     global grads
#     if PATTERN == 'Clean':
#         filename = f"{prefix}Uncertainty/{model}/{PATTERN.upper()}/{dataset}_uncertainty.tsv"
#     elif PATTERN == 'human':
#         if pt == 'A':
#             filename = f"{prefix}Uncertainty/{model}/{PATTERN}/{pt}/{nk}/{dataset}_uncertainty.tsv"
#         else:
#             filename = f"{prefix}Uncertainty/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}_uncertainty.tsv"
#     else: ## All else
#         filename = f"{prefix}Uncertainty/{model}/{PATTERN}/{nk}/{lvl}/{dataset}_uncertainty.tsv"
    
#     if dataset == 'hatexplain' or dataset == 'esnli':
#         n_classes = 3
#     else:
#         n_classes = 2
        
#     if model == "opt":
#         NoiseFolder =  "OPTNoise"
#     else:
#         NoiseFolder =  "Noise" #"OPTNoise" #Noise

#     try:
#         df_unc = pd.read_csv(filename, index_col=0)

#         # if (dataset == 'hatexplain') and( temp_fix == True): ############ CATCH
#         #     print("REMOVING DATAPOINTS 304,407,932 DUE TO CORRUPTION")
#         #     df.drop(index=[304,407,932], inplace = True)  
        
#     except (FileNotFoundError, ValueError) as error:
#         df_unc = pd.DataFrame([])

#     if dataset == 'SST-2':                
#         anno_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"annotations"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="anno").map(lambda x: np.array([float(y) for y in x.split()]))
#     elif dataset == 'esnli':
#         out = pd.DataFrame([load_from_disk(f"./Data/Noise/esnli")[f"annotation_1"],load_from_disk(f"./Data/Noise/esnli")[f"annotation_2"]]).T
#         out.index = load_from_disk(f"./Data/Noise/esnli")["index"]
#         out['anno'] = out.apply(lambda x: np.array(x[0] + [0.0] + x[1]), axis=1)
#         anno_pattern = out['anno']
#     else:
#         anno_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"annotations"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="anno").map(lambda x: np.array([float(y) for y in x]))
            
#     label_pattern =  pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"label"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="label").map(lambda x: x*2 - 1)

#     anno_pattern *= label_pattern
#     anno_pattern.name = 'anno'
    
#     if PATTERN == 'human':
#         if pt == 'A':
#             perturb_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"{PATTERN}-{pt}_PATTERN"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="perturbation")
#         else:
#             perturb_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"{PATTERN}-{pt}_PATTERN_{lvl}"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="perturbation")
#     elif PATTERN == 'gradient':
#         perturb_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"{PATTERN}-{model}_PATTERN_{lvl}"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="perturbation")
#     elif PATTERN == 'random': # random
#         perturb_pattern = pd.Series(load_from_disk(f"./Data/{NoiseFolder}/{dataset}")[f"{PATTERN}_PATTERN_{lvl}"], index=load_from_disk(f"./Data/{NoiseFolder}/{dataset}")["index"], name="perturbation")
    
#     if PATTERN != "Clean":
#         perturb_pattern.map(lambda x: [int(literal_eval(y)) for y in x.split()])
        
#     # if dataset == 'hatexplain' and temp_fix: ############ CATCH
#     #     print("REMOVING DATAPOINTS 304,407,932 DUE TO CORRUPTION")
#     #     anno_pattern.drop(index=[304,407,932], inplace = True) 
#     #     label_pattern.drop(index=[304,407,932], inplace = True)
#     #     if PATTERN != "Clean":
#     #         perturb_pattern.drop(index=[304,407,932], inplace = True)  
        
#     all_annos_pattern= anno_pattern.map(lambda x: np.where(x>0, 1, 0))
#     majority_annos_pattern = anno_pattern.map(lambda x: np.where(x>0.5, 1, 0))
#     all_annos_pattern.name = 'anno'
#     majority_annos_pattern.name = 'anno'

#     grad_dfs = []

#     for k in grads:
#         try:
#             og_filename = f"{prefix}Saliency/Updated/{model}/{dataset}-{k}.tsv"
#             if not os.path.exists(og_filename):
#                 og_filename = og_filename.replace("Updated", "Clean")
#             df_og = pd.read_csv(og_filename, index_col=0, sep = '\t',quoting=3)
        
#             if PATTERN != 'Clean':
#                 if PATTERN == 'human':
#                     if pt == 'A':
#                         filename = f"{prefix}Saliency/Updated/{model}/{PATTERN}/{pt}/{nk}/{dataset}/{k}/{k}.tsv"
#                         if not os.path.exists(filename):
#                             filename = filename.replace("Updated", "Clean")
#                         df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
#                     else:
#                         filename = f"{prefix}Saliency/Updated/{model}/{PATTERN}/{pt}/{nk}/{lvl}/{dataset}/{k}/{k}.tsv"
#                         if not os.path.exists(filename):
#                             filename = filename.replace("Updated", "Clean")
#                         df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
#                 else:
#                     filename = f"{prefix}Saliency/Updated/{model}/{PATTERN}/{nk}/{lvl}/{dataset}/{k}/{k}.tsv"
#                     if not os.path.exists(filename):
#                         filename = filename.replace("Updated", "Clean")
#                     df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
                    
#             else:
#                 filename = f"{prefix}Saliency/Updated/{model}/{dataset}-{k}.tsv"
#                 if not os.path.exists(filename):
#                     filename = filename.replace("Updated", "Clean")
#                 df = pd.read_csv(filename, index_col=0, sep = '\t',quoting=3)
            
#             # if dataset == 'hatexplain' and temp_fix:
#             #     df.drop(index=[304,407,932], inplace= True)
#             #     df_og.drop(index=[304,407,932], inplace= True)
            
#             df['attribution_scores'] = df['attribution_scores'].map(convert_to_list)
#             df_og['attribution_scores'] = df_og['attribution_scores'].map(convert_to_list)
            
#             ## Get correlation to original saliency map
#             if PATTERN == 'Clean':
#                 df[f'{k}_og_corr'] = 1.0
#             else:
#                 df[f'{k}_og_corr'] = df[['attribution_scores']].merge(df_og[['attribution_scores']], on ='id', how = 'inner').dropna().apply(lambda x: get_correlation(x.attribution_scores_x, x.attribution_scores_y, temp_fix=True, extra_data=[dataset,PATTERN, nk, k,lvl, x.index]), axis=1)
                
#             if PATTERN == 'Clean':
#                 ## Get correlation to perturbation map
#                 df[f'{k}_noise_corr'] = np.nan ## No perturbation in Clean data
#                 df[f'{k}_noise_map'] = np.nan
#             else:
#                 ## Get correlation to perturbation map
#                 df[f'{k}_noise_corr']  = df[['attribution_scores']].merge(perturb_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_correlation(x.attribution_scores, x.perturbation, temp_fix=True, extra_data=[dataset,PATTERN, nk, k, lvl, x.index]), axis=1)
#                 df[f'{k}_noise_map'] = df[['attribution_scores']].merge(perturb_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_map( x.perturbation, x.attribution_scores, temp_fix=True, extra_data=[dataset,PATTERN, nk,k, lvl, x.index]), axis=1)                
            
#             ## Get correlation to humman annotation
#             df[f'{k}_anno_corr'] = df[['attribution_scores']].merge(anno_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_correlation(x.attribution_scores, x.anno, temp_fix=True, extra_data=[dataset,PATTERN, nk,k, lvl, x.index]), axis=1) #temp_fix=False, extra_data=None):
            
#             ## Get correlation to humman annotation
#             df[f'{k}_all_anno_map'] = df[['attribution_scores']].merge(all_annos_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_map( x.anno, x.attribution_scores, temp_fix=True, extra_data=[dataset,PATTERN, nk,k, lvl, x.index]), axis=1)
            
#             df[f'{k}_majo_anno_map'] = df[['attribution_scores']].merge(majority_annos_pattern, left_index=True, right_index=True, how = 'inner').dropna().apply(lambda x: get_map( x.anno, x.attribution_scores, temp_fix=True, extra_data=[dataset,PATTERN, nk,k, lvl, x.index]), axis=1)
            
#             df[f'{k}_complexity'] = df['attribution_scores'].map(get_entropy_complexity)
            
#             df[f'{k}_zvd'] = df['attribution_scores'].map(get_zero_vd)
            
#         except (FileNotFoundError, ValueError) as error:
#             df = pd.DataFrame([])
#             print(f"Missing filename: {filename}")
#             print(error)
            
#         grad_dfs.append(df)
            
#     return pd.concat([df_unc] + grad_dfs, axis=1)

def __main__():
    exit()