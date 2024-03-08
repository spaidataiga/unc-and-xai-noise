from scipy.stats import entropy
from gradient_utils import *
from model_utils import *
import time
import numpy as np
import os
import pandas as pd

def run_and_write_gradients(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL):
    # Eval!
    print(f"***** Running evaluation*****")
    t0 = time.time()
    
    ################################################## LOAD GRADIENT METHODS ######################################################


    gradient_methods = prepare_for_gradients(modelwrapper)
    
    ###############################################################################################################################
    
    ###############################################  GO THROUGH TEST EXAMPLES #######################################################

    preds = None
    out_label_ids = None
    ids = []
    entropies = []


    # captum doesn't support batching well, so interpreting per instance
    for batch in dataloader:
        
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
        }

        if MODEL == 'bert' and DATA == 'esnli':
            with torch.no_grad():
                logit = modelwrapper.model(batch['input_ids'].to(device), batch['attention_mask'].to(device)).logits
        else:
            with torch.no_grad():
                logit = modelwrapper.model(**inputs).logits

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
            
            


    directory = f"../../Saliency/RAW/{MODEL}/{DOWNSTREAM_FOLDER}/{DATA}/"
    if not os.path.exists(directory):
    # If it doesn't exist, create it
        os.makedirs(directory)
    # visualize results
    for k in gradient_methods.keys():
        visualize_text_pred(gradient_methods[k]['vis_list'], ids, entropies, f"{directory}{k}")

        output_interpret_feature_file = f"{directory}{k}.csv"

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
    
    
def run_and_write_uncertainty(modelwrapper, eval_dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL, FWD_PASSES=100):
        
    if DATA == 'SST-2' or DATA == 'SemEval':
        n_classes = 2
    else: #esnli, hatexplain
        n_classes = 3
        
    
    dropout_predictions = np.empty((0, eval_dataloader.dataset.num_rows, n_classes)) ### 0 x # samples x # classes, empty container to hold all dropout predictions


    ###############################################  GO THROUGH TEST EXAMPLES #######################################################

    # Eval!
    print("***** Running evaluation *****")
    t0 = time.time()


    for i in range(int(FWD_PASSES)): # Run multiple times for each instance
        
        preds = None
        out_label_ids = None
        ids = []

        for batch in eval_dataloader: # Go through training instancer
            
            # if (batch["index"] == 459) and (DATA == 'esnli'):
            #     continue
            
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }

            
            with torch.no_grad():
                logit = modelwrapper.model(**inputs).logits

            logit = torch.nn.functional.softmax(logit, dim=1) 
            
            if preds is None:
                preds = logit.detach().cpu().numpy()
                if 'labels' in inputs:
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logit.detach().cpu().numpy(), axis=0)
                if 'labels' in inputs:
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                    
            ids.append(batch['index'].item())

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
    
    directory = f"../../Uncertainty/{MODEL}/{DOWNSTREAM_FOLDER}"
    if not os.path.exists(directory):
    # If it doesn't exist, create it
        os.makedirs(directory)
          
    results = {'entropy': entropy, 'mutual_info': mutual_info}

    for m in range(mean.shape[1]):
        results[f"mean_{m}"] = mean[:,m]
        
    for m in range(variance.shape[1]):
        results[f"variance_{m}"] = variance[:,m]
    
    out = pd.DataFrame(results, index= ids)
    out.to_csv(f"{directory}{DATA}_uncertainty.tsv")
    
    
def run_and_write_gradients_SD(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL):
    # Eval!
    print(f"***** Running evaluation*****")
    t0 = time.time()
    
    ################################################## LOAD GRADIENT METHODS ######################################################


    gradient_methods = prepare_for_gradients(modelwrapper, all_classes=True)
    
    ###############################################################################################################################
    
    
    if DATA == 'SST-2' or DATA == 'SemEval':
        n_classes = 2
    else: #esnli, hatexplain
        n_classes = 3   
    ###############################################  GO THROUGH TEST EXAMPLES #######################################################

    preds = None
    out_label_ids = None
    ids = []
    entropies = []
    for k in gradient_methods.keys():
        for c in range(n_classes): #initialize list
            gradient_methods[k]['vis_list'][c] = []


    # captum doesn't support batching well, so interpreting per instance
    for batch in dataloader:
        
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
        }

        
        with torch.no_grad():
            logit = modelwrapper.model(**inputs).logits

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
                
                for c in range(n_classes):

                    attributions, delta = gradient_methods[k]["model"].attribute(input_embedding,
                                                                                baselines=ref_input_ids,
                                                                                target = c,
                                                                                **gradient_methods[k]["config"])
                    attributions = summarize_attributions(attributions)


                    vis = add_attributions_to_visualizer_pred(
                                        attributions,
                                        tokens[:nz],
                                        logit_for_vis,
                                        pred_idx,
                                        batch['labels'],
                                        delta) 
                    
                    gradient_methods[k]['vis_list'][c].append(vis)  
                
                
            else:   
                for c in range(n_classes):   
                    attributions = gradient_methods[k]["model"].attribute(  inputs=input_embedding,
                                                                            target=c,
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
                    
                    gradient_methods[k]['vis_list'][c].append(vis) 
            
              
            
            for token, attr in zip(tokens[1:-1], attributions[1:-1]):
                gradient_methods[k]['attr_token_dic'][token].append(attr) 
            
            

    for c in range(n_classes):
        directory = f"../../Saliency/{DOWNSTREAM_FOLDER}/{MODEL}/Classes/{DATA}/{c}/"
        if not os.path.exists(directory):
        # If it doesn't exist, create it
            os.makedirs(directory)
        # visualize results
        for k in gradient_methods.keys():
            visualize_text_pred(gradient_methods[k]['vis_list'][c], ids, entropies, f"{directory}{k}")

            output_interpret_feature_file = f"{directory}{k}.csv"

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
    
def get_specific_gradients(modelwrapper, dataloader, tokenizer, device, MODEL, k, i):
    # Eval!
    print(f"***** Running evaluation*****")
    t0 = time.time()
    
    ################################################## LOAD GRADIENT METHODS ######################################################
    lig = False
    ixg = False
    sg = False
    gbp = False
    if k == "InputXGrad":
        ixg = True
    elif k == "SmoothGrad":
        sg = True
    elif k == "GuidedBP":
        gbp = True
    elif k == "IntegGrad":
        lig = True
        
    gradient_methods = prepare_for_gradients(modelwrapper, lig, ixg, sg, gbp)
    
    ###############################################################################################################################
    
    ###############################################  GO THROUGH TEST EXAMPLES #######################################################

    preds = None
    out_label_ids = None
    ids = None
    entropies = None

    # iter(trainloader).next()
    
    batch = next(iter(dataloader))
    # captum doesn't support batching well, so interpreting per instance
    # for batch in dataloader:
        
    inputs = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
    }

    
    with torch.no_grad():
        logit = modelwrapper.model(**inputs).logits

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

    ids = batch['index'].item() # Keep track of id for matching later
    # print(ids)
    
    pred_idx = torch.argmax(logit)
    logit_for_vis = logit[0][pred_idx]
    
    entropies = entropy(logit.detach().cpu().numpy()[0]) ### calculate H, axis = 1 to account for batches

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
        
        gradient_methods[k]['vis_list'] = vis                  
        
        for token, attr in zip(tokens[1:-1], attributions[1:-1]):
            gradient_methods[k]['attr_token_dic'][token].append(attr) 
        
    return [gradient_methods[k]['vis_list'], ids, entropies]

def __main__():
    exit()