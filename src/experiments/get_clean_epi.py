
import torch
import numpy as np
from ..utils.gradient_utils import *
from ..utils.model_utils import *
import time
import pandas as pd
import sys
import os


os.environ["TORCH_USE_CUDA_DSA"] = "1" ### HELPS DEBUGGING
FWD_PASSES = 100 # In original article: https://arxiv.org/abs/1506.02142

if __name__  == "__main__":
    ######################################## DECLARE DATA TO USE AND MODEL TYPE ##############################################

    MODEL = sys.argv[1].lower()
    DATA =  sys.argv[2]

    MODEL_BASE, text_label, is_lower, MAX_LENGTH, idx_label, num_labels = get_model_details(MODEL,DATA)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    #############################################################################################################################


    eval_dataloader, model, _ = get_model_data(MODEL_BASE, DATA, device, sequential=True) #max_size=100?

    #### Wrapper with custom forward function required for Captum
    modelwrapper = ModelWrapper(model, device)
    modelwrapper.to(device)
    modelwrapper.eval()
    modelwrapper.enable_dropout() ##### Turn on dropout for MC
    modelwrapper.zero_grad()

    dropout_predictions = np.empty((0, eval_dataloader.dataset.num_rows, num_labels)) ### 0 x # samples x # classes, empty container to hold all dropout predictions


    ###############################################  GO THROUGH TEST EXAMPLES #######################################################

    # Eval!
    print("***** Running evaluation *****")
    t0 = time.time()

    for i in range(FWD_PASSES): # Run multiple times for each instance
        
        preds = None
        out_label_ids = None
        ids = []

        for batch in eval_dataloader: # Go through training instancer
            
            # if batch['index'].item() in [459,570, 571] and DATA == "esnli":
            #     continue
            
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }

            
            try:
                with torch.no_grad():
                    logit = model(**inputs).logits

                logit = torch.nn.functional.softmax(logit, dim=1) 
                
                if preds is None:
                    preds = logit.detach().cpu().numpy()
                    if 'labels' in inputs:
                        out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logit.detach().cpu().numpy(), axis=0)
                    if 'labels' in inputs:
                        out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                        
                ids.append(batch['index'].detach().cpu())
            except RuntimeError:
                print("CATCHED!!!", batch['index'].item())
                torch.cuda.empty_cache() 
                del inputs
                # continue

        dropout_predictions = np.vstack((dropout_predictions,
                                            preds[np.newaxis, :, :]))
            
            
    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes 
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                            axis=-1), axis=0)  # shape (n_samples,)


    t1 = time.time()

    print(f"It took {t1-t0} seconds to run all datapoints")

    ################################# ASSUMES ONLY TWO CLASSES, IF MORE, cHANGE #######################################################

    results = {'entropy': entropy, 'mutual_info': mutual_info}

    #'mean_0': mean[:,0], 'mean_1': mean[:,1], 'variance_0': variance[:,0], 'variance_1': variance[:,1], 
    for m in range(mean.shape[1]):
        results[f"mean_{m}"] = mean[:,m]
        
    for m in range(variance.shape[1]):
        results[f"variance_{m}"] = variance[:,m]

    # results = {'mean_0': mean[:,0], 'mean_1': mean[:,1], 'variance_0': variance[:,0], 'variance_1': variance[:,1], 'entropy': entropy, 'mutual_info': mutual_info}
    out = pd.DataFrame(results, index= ids)

    directory = f"../../Uncertainty/{MODEL}/CLEAN/"
    if not os.path.exists(directory):
    # If it doesn't exist, create it
        os.makedirs(directory)
            
    out.to_csv(f"{directory}{DATA}_uncertainty.tsv")