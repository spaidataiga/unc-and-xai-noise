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
import json
from pdb import set_trace

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


if __name__ == "__main__":
    ######################################## DECLARE DATA TO USE AND MODEL TYPE ##############################################
    MODEL = sys.argv[1].lower()
    DATA = sys.argv[2]

    directory = "Hotflip/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    IN_ORDER = True
    NUM_SAMPLES = None #None if entire dataset

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


    vanilla = Saliency(modelwrapper)

    ###############################################  GO THROUGH TEST EXAMPLES #######################################################

    # Eval!
    print("***** Running evaluation *****")
    t0 = time.time()

    preds = None
    out_label_ids = None

    ids = []
    hotflip_out = []
    all_tokens = []

    # captum doesn't support batching well, so interpreting per instance
    for batch in eval_dataloader:
        
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

            
            nz = (batch['attention_mask']!=0).sum()
            
            if 'gpt2' in MODEL:
                input_ids  = inputs['input_ids'][:,-nz:] #### Padding to the left
            else:
                input_ids = inputs['input_ids'][:,:nz] 
                
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy().tolist())

            
            pred_idx = torch.argmax(logit)
            logit_for_vis = logit[0][pred_idx]
                
            input_embedding = modelwrapper.get_embedding(input_ids)
            
            attributions = vanilla.attribute(input_embedding, target=pred_idx.item())
                
            hotflip_grad = np.einsum('bld,kd->blk', attributions.detach().cpu(), model.get_input_embeddings().weight.data.cpu()) #create lookup table
            
            attributions = []
            for i,j in enumerate(input_ids[0]):
                og = hotflip_grad[0,i,j]
                candidates = hotflip_grad[0,i,:] - og
                attributions.append(np.mean(candidates)) #find average change in grad if token is switched
            
            hotflip_out.append(attributions)
            all_tokens.append(tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy().tolist()))
            ids.append(batch['index'].item())
        except RuntimeError:
            print(batch['index'].item())
        



    full = load_from_disk(f"./Data/Clean/{DATA}")['test'] # text form

    if type(text_label) == list:
        orig_data = []
        for l in range(len(full[text_label[0]])):
            line = ""
            for t in text_label:
                line += full[t][l]
                line += " | "
            line = line[:-3]
            orig_data.append(line)
    else:
        orig_data = full[text_label]
        
    del full

    raw_input_ids = []
    word_attributions = []
    word2attributions = defaultdict(list)
    pred_token_dic = defaultdict(list)
    rankings = {}
    
    counter = 0
    for idx, sentence, score, orig_text in zip(ids, all_tokens, hotflip_out, orig_data):
        bert_tokens, attr_list = remove_cls_and_bert(sentence, score, already_tokenized = True)
        orig2bert, idx2token = match_bert_token_to_original(bert_tokens, orig_text, model)
        raw_input_ids.append(list(idx2token.values()))
        new_attr_dict = make_new_attr_score(orig2bert, attr_list)
        attributions = list(new_attr_dict.values())
        rankings[idx] = list(np.argsort(attributions)) #Get negative sign so in descending order?
        counter += 1


    with open(f"{directory}{MODEL}-{DATA}.json", "w") as outfile:
        # json_data refers to the above JSON
        json.dump(rankings, outfile, default=np_encoder)
