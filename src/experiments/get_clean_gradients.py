"""
This is source code to collect all the desired saliency maps for a model output, regardless of the model.
"""


import torch
from captum.attr._utils.visualization import *
import numpy as np
from ..utils.gradient_utils import *
from ..utils.model_utils import *
import time
import os
from scipy.stats import entropy
import sys


WRITE = True
IN_ORDER = True
DOWNSTREAM_FOLDER = "RAW" 
NUM_SAMPLES = None #None if entire dataset


if __name__ == "__main__":
    ######################################## DECLARE DATA TO USE AND MODEL TYPE ##############################################
    MODEL = sys.argv[1].lower()
    DATA = sys.argv[2]

    MODEL_BASE, text_label, is_lower, MAX_LENGTH, idx_label, num_labels = get_model_details(MODEL,DATA)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    #############################################################################################################################

    eval_dataloader, model, tokenizer = get_model_data(MODEL_BASE, DATA, device, max_size=NUM_SAMPLES, sequential=IN_ORDER)

    #### Wrapper with custom forward function required for Captum
    modelwrapper = ModelWrapper(model, device)
    modelwrapper.to(device)
    modelwrapper.eval()
    modelwrapper.zero_grad()

    ################################################## LOAD GRADIENT METHODS ######################################################

    gradient_methods = prepare_for_gradients(modelwrapper)

    ###############################################  GO THROUGH TEST EXAMPLES #######################################################

    # Eval!
    print("***** Running evaluation *****")
    t0 = time.time()

    preds = None
    out_label_ids = None
    ids = []
    entropies = []


    # captum doesn't support batching well, so interpreting per instance
    for batch in eval_dataloader:
        
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
        }

        
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

        
        nz = (batch['attention_mask']!=0).sum()
        
        if 'gpt2' in MODEL:
            input_ids  = inputs['input_ids'][:,-nz:] #### Padding to the left
        else:
            input_ids = inputs['input_ids'][:,:nz] 
            
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy().tolist())

        ids.append(batch['index'].item()) # Keep track of id for matching later
        
        pred_idx = torch.argmax(logit)
        logit_for_vis = logit[0][pred_idx]
        
        entropies.append(entropy(logit.detach().cpu().numpy()[0])) ### calculate H, axis = 1 to account for batches

        input_embedding = modelwrapper.get_embedding(input_ids)
        
        ########################### CREATE ALL SALIENCY MAPS #########################################
        for k in gradient_methods.keys():
            
            if k == 'IntegGrad': #### Requires different
                
                if 'gpt2' in MODEL:
                    ref_input_ids = batch['ref_ids'].to(device)[:,-nz:]
                else:
                    ref_input_ids = batch['ref_ids'].to(device)[:,:nz]

                
                ref_input_ids = modelwrapper.get_embedding(ref_input_ids)

                attributions, delta = gradient_methods[k]["model"].attribute(input_embedding,
                                                                            baselines=ref_input_ids,
                                                                            target = pred_idx.item(),
                                                                            **gradient_methods[k]["config"])
                
                
            else:   
                attributions = gradient_methods[k]["model"].attribute(  inputs=input_embedding,
                                                                        target=pred_idx.item(),
                                                                        **gradient_methods[k]["config"])
                delta = None
            

            attributions = summarize_attributions(attributions)


            vis = add_attributions_to_visualizer_pred(
                                attributions,
                                tokens[:nz],
                                logit_for_vis,
                                pred_idx,
                                batch['labels'],
                                delta) 
            
            gradient_methods[k]['vis_list'].append(vis)                    
            
            for token, attr in zip(tokens[1:-1], attributions[1:-1]):
                gradient_methods[k]['attr_token_dic'][token].append(attr) 
            
            

    if WRITE:
        directory = f"../../Saliency/{DOWNSTREAM_FOLDER}/{MODEL}/"
        if not os.path.exists(directory):
        # If it doesn't exist, create it
            os.makedirs(directory)
        # visualize results
        for k in gradient_methods.keys():
            visualize_text_pred(gradient_methods[k]['vis_list'], ids, entropies, f"{directory}{DATA}-{k}")

            output_interpret_feature_file = f"{directory}{DATA}-{k}.csv"

            attr_avg = ((np.average(values), tks) for tks, values in gradient_methods[k]['attr_token_dic'].items())
            print("***** Interpret feature file saved: {} *****".format(output_interpret_feature_file))
            with open(output_interpret_feature_file, "w") as writer:
                for idx, (avg, tks)  in enumerate(sorted(attr_avg, reverse=True)):
                    writer.write("{}\t{}\t{}\t{}\t{}\n".format(
                        idx, tks, len(gradient_methods[k]['attr_token_dic'][tks]),
                        np.average(gradient_methods[k]['attr_token_dic'][tks]),
                        np.std(gradient_methods[k]['attr_token_dic'][tks])) 
                    )

    t1 = time.time()

    print(f"It took {t1-t0} seconds to run all datapoints")
