import torch
from captum.attr._utils.visualization import *
import numpy as np
from captum.attr import LayerIntegratedGradients, InputXGradient, Saliency, NoiseTunnel, GuidedBackprop
from collections import defaultdict
from sklearn.metrics import f1_score


"""Code referenced from
https://github.com/sweetpeach/hummingbird/blob/main/code/model/captum_label.py
https://github.com/sweetpeach/hummingbird/blob/main/code/model/utils.py
https://github.com/mt-upc/transformer-contributions/blob/main/src/contributions.py
https://github.com/mt-upc/transformer-contributions/blob/main/src/utils_contributions.py
https://github.com/mt-upc/transformer-contributions/blob/main/Text_classification.ipynb
"""

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions.detach().cpu().numpy()


def acc_and_f1(preds, labels):
    acc = (preds == labels).mean()

    label_set = list(set(labels))
    new_labels, new_preds =  [], []
    unmatched_label_prediction_cnt = 0
    for l,p in zip(labels,preds):
        if p not in label_set:
            unmatched_label_prediction_cnt += 1
        else:
            new_preds.append(p)
            new_labels.append(l)
    if unmatched_label_prediction_cnt > 0:
        # from pdb import set_trace; set_trace()
        f1 = f1_score(y_true=new_labels, y_pred=new_preds, average='macro')
    else:
        f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "unmatched_label_prediction_cnt": unmatched_label_prediction_cnt,
        "cnt": len(preds)
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)

    return acc_and_f1(preds, labels)


def add_attributions_to_visualizer_pred(attributions, tokens, pred, pred_ind, true_class, delta=0):
    # storing couple samples in an array for visualization purposes

    vis = VisualizationDataRecord(
                            attributions, #word_attributions
                            pred, #pred_prob
                            pred_ind, #pred_class
                            true_class, #true_class ############################################ I HAE NO IDEA WHAT THIS SHOULD BE
                            100, #true_class, #attr_class in docs #1 in OG code ############################################ I HAE NO IDEA WHAT THIS SHOULD BE
                            attributions.sum(), #attr_score
                            tokens[:len(attributions)],  #raw_input_ds
                            delta) #convergence_score
    return vis

def prepare_for_gradients(modelwrapper, lig = True, ixg=True, sg=True, gbp = True, ggc = False, all_classes=False):
    
    
    gradient_methods = {}
    
    ### First create models and their configuration rules for attributions

    if lig:
        if modelwrapper.model.config.model_type == 'bert':
            gradient_methods['IntegGrad'] = { "model" : LayerIntegratedGradients(modelwrapper, modelwrapper.model.bert.embeddings)}
        elif modelwrapper.model.config.model_type == 'electra':
            gradient_methods['IntegGrad'] = { "model" : LayerIntegratedGradients(modelwrapper, modelwrapper.model.electra.embeddings)}
        elif modelwrapper.model.config.model_type == 'roberta':
            gradient_methods['IntegGrad'] = { "model" : LayerIntegratedGradients(modelwrapper, modelwrapper.model.roberta.embeddings)}
        elif modelwrapper.model.model.decoder.config.model_type == 'opt':
            gradient_methods['IntegGrad'] = { "model" : LayerIntegratedGradients(modelwrapper, modelwrapper.model.model.decoder.project_in)}
        else: #gpt2 ### IntegGrad for GPT2, don't use layerintegrated gradients? See https://captum.ai/tutorials/Bert_SQUAD_Interpret and https://github.com/huggingface/transformers/issues/1458
            gradient_methods['IntegGrad'] = { "model" : LayerIntegratedGradients(modelwrapper, modelwrapper.model.transformer.drop)}
            #gradient_methods['IntegGrad'] = { "model" : LayerIntegratedGradients(modelwrapper, [modelwrapper.model.transformer.wte, modelwrapper.model.transformer.wpe])}
        gradient_methods['IntegGrad']["config"] = { 'n_steps':100,
                                                    'return_convergence_delta':True,
                                                    'internal_batch_size':10}
            
    if ixg:    
        gradient_methods['InputXGrad'] = { "model" : InputXGradient(modelwrapper)}
        gradient_methods['InputXGrad']["config"] = {}
        
    if sg:
        gradient_methods['SmoothGrad'] = { "model" : NoiseTunnel(Saliency(modelwrapper))} # chose Vanilla gradients as base
        gradient_methods['SmoothGrad']["config"] = {'nt_type':'smoothgrad',
                                                    'nt_samples':10} # following original paper    
    if gbp:
        gradient_methods['GuidedBP'] = { "model" : GuidedBackprop(modelwrapper)}
        gradient_methods['GuidedBP']["config"]  = {}
        
    # if ggc:
    #     gradient_methods['GuidedGradCAM'] = { "model" : GuidedGradCam(modelwrapper, modelwrapper.model.bert.encoder.layer[11])}
    #     gradient_methods['GuidedGradCAM']["config"] = {'interpolate_mode':'nearest'} # bilinear in original paper but requires 4D input


    for k in gradient_methods.keys():
        gradient_methods[k]['attr_token_dic'] = defaultdict(list)
        if all_classes:
            gradient_methods[k]['vis_list'] = {}
        else:
            gradient_methods[k]['vis_list'] = []
        
    return gradient_methods

def visualize_text_pred(datarecords, ids, entropies, filename) -> None:
    print("visualize prediction text to ", filename)
    dom = []
    rows = [
        "<table width: 100%>"
        "<th>Index</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]

    preds_ids,label_ids = [], []
    tsv_file = filename+".tsv"
    html_file = filename+".html"
    tsv_rows = []
    for i,datarecord in enumerate(datarecords):
        color = ''
        # if int(datarecord.pred_class) == int(datarecord.true_class):
        #     color = "bgcolor=#ccccff"
        # else:
        #     color = "bgcolor=#ffb3b3"
        color = "bgcolor=#ccccff"


        rows.append(
            "".join(
                [
                    "<tr {}>".format(color),
                    # format_classname(datarecord.true_class),
                    # format_classname(datarecord.target_class),
                    format_classname(ids[i]),
                    format_classname(
                        "{0} ({1:.2f}) ({2:.6f})".format(
                            int(datarecord.pred_class), float(datarecord.pred_prob), float(entropies[i])
                        )
                    ),
                    format_classname("{0:.2f}".format(float(datarecord.true_class))),
                    # format_classname("{0:.2f}".format(float(datarecord.attr_score))),
                    format_word_importances(
                        datarecord.raw_input_ids, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )
        prediction = str(int(datarecord.pred_class))
        truth = str(int(datarecord.true_class))
        prediction_prob = "{:.2f}".format(float(datarecord.pred_prob))
        the_raw_input = datarecord.raw_input_ids
        word_attr_scores = datarecord.word_attributions
        temp = []
        importance_list = []
        for word, importance in zip(the_raw_input, word_attr_scores[:len(the_raw_input)]):
            word = format_special_tokens(word)
            temp_str = word #+ " ({:.2f})".format(importance)
            temp.append(temp_str)
            importance_list.append("{:.6f}".format(importance))
        text = " ".join(temp)
        attr_list = " ".join(importance_list)
        tsv_row = [str(ids[i]), prediction, truth, prediction_prob, str(entropies[i]), text, attr_list]
        tsv_rows.append(tsv_row)

        preds_ids.append(int(datarecord.pred_class))
        label_ids.append(int(datarecord.true_class))

    
    with open(tsv_file, 'w') as writer:
        header = "id\tpred_class\ttrue_class\tpred_prob\tpred_entropy\traw_input\tattribution_scores\n"
        writer.write(header)
        for row in tsv_rows:
            to_be_written = "\t".join(row)
            writer.write(to_be_written+"\n")

    result = compute_metrics(np.array(preds_ids), np.array(label_ids))
    dom.append("<p>Samples: {}, {}</p>".format(len(preds_ids), result))

    dom.append("".join(rows))
    dom.append("</table>")

    html = HTML("".join(dom))
    print("done")
    # print(html.data)
    with open(html_file, 'w') as f:
        f.write(html.data)
    print("finish writing to ", filename)
    
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

def remove_cls_and_bert(sentence, attr_score_list, already_tokenized=False):
    if not already_tokenized:
        tokens = sentence.split(" ")
        attr_scores = attr_score_list.split(" ")
    else:
        tokens = sentence
        attr_scores = attr_score_list  
    new_tokens = tokens
    new_attrs = attr_scores
    if tokens[0].lower() == "[cls]": # Bert and Electra
        if tokens[len(tokens)-1].lower() == "[sep]":
            new_tokens = tokens[1:len(tokens)-1]
            new_attrs = attr_scores[1:len(tokens)-1]
    if tokens[0].lower() == "#s": # Roberta
        if tokens[len(tokens)-1].lower() == "#/s":
            new_tokens = tokens[1:len(tokens)-1]
            new_attrs = attr_scores[1:len(tokens)-1]
    if len(new_tokens) != len(new_attrs):
        print("error here")
    return new_tokens, new_attrs

def match_bert_token_to_original(bert_tokens, orig_text, model='bert'):
    raw_orig = orig_text.split(" ")
    orig = []
    for word in raw_orig:
        if word != "":
            orig.append(word.strip())
    bert_idx = 0
    last_bert_idx = 1
    orig_idx = 0
    
    orig_to_bert_mapping = {}
    orig_idx2token = {}

    if model in ['bert', 'electra']: #an art ##ful --> An artful
        while bert_idx < len(bert_tokens) and orig_idx < len(orig):

            current_orig = orig[orig_idx]
            current_bert = bert_tokens[bert_idx]
            orig_to_bert_mapping[orig_idx] = [bert_idx]
            
            bert_idx += 1
            orig_idx2token[orig_idx] = current_orig
            if current_bert != current_orig:			
                combined = current_bert
                last_bert_idx = bert_idx
                while last_bert_idx < len(bert_tokens):
                    next_part = bert_tokens[last_bert_idx].replace("##","").strip()				
                    combined += next_part
                    orig_to_bert_mapping[orig_idx].append(last_bert_idx)
                    # if current_orig == "well-established":
                    # 	print("combined: ", combined)
                    if combined == current_orig:					
                        bert_idx = last_bert_idx + 1
                        break
                    else:
                        last_bert_idx += 1

            orig_idx += 1
    else: # roberta  #Ġan Ġart ful --> An artful
        while bert_idx < len(bert_tokens) and orig_idx < len(orig):

            current_orig = orig[orig_idx]
            current_bert = bert_tokens[bert_idx].replace("Ġ","").strip() # Remove any Ġ at the start of word
            orig_to_bert_mapping[orig_idx] = [bert_idx]
            
            bert_idx += 1
            orig_idx2token[orig_idx] = current_orig
            if current_bert != current_orig:			
                combined = current_bert
                last_bert_idx = bert_idx
                while last_bert_idx < len(bert_tokens):
                    next_part = bert_tokens[last_bert_idx]			
                    combined += next_part
                    orig_to_bert_mapping[orig_idx].append(last_bert_idx)
                    # if current_orig == "well-established":
                    # 	print("combined: ", combined)
                    if combined == current_orig:					
                        bert_idx = last_bert_idx + 1
                        break
                    else:
                        last_bert_idx += 1

            orig_idx += 1
    return orig_to_bert_mapping, orig_idx2token

def make_new_attr_score(orig2bert, bert_attribution_scores):
	new_score = {}
	for key, value in orig2bert.items():
		the_list = []
		for bert_idx in value:
			the_list.append(bert_attribution_scores[bert_idx])
		np_list = np.array(the_list).astype(float)
		
		attr_avg = np.mean(np_list)
		new_score[key] = attr_avg
	return new_score

def process_bert(bert_data, orig_data, model='bert', token_type=None):

    new_data_record = list(bert_data[['pred_class', 'true_class', 'pred_prob', 'pred_entropy']].transpose().to_dict().values())

    bert_sentences = bert_data["raw_input"].values
    bert_attribution_scores = bert_data["attribution_scores"].values
    pred_probs = bert_data["pred_prob"].values
    
    if token_type:
        g = {'TOKEN': f'[{token_type.upper()}]'}

    word2attributions = defaultdict(list)
    pred_token_dic = defaultdict(list)

    counter = 0
    for sentence, score, orig_text, prob in zip(bert_sentences, bert_attribution_scores, orig_data, pred_probs):
        if token_type:
            orig_text = orig_text.format(**g) ## replace token filler with token used
        bert_tokens, attr_list = remove_cls_and_bert(sentence, score)
        orig2bert, idx2token = match_bert_token_to_original(bert_tokens, orig_text, model)
        new_data_record[counter]['raw_input_ids'] = list(idx2token.values())
        new_attr_dict = make_new_attr_score(orig2bert, attr_list)
        new_data_record[counter]['word_attributions'] = list(new_attr_dict.values())
        for idx, score in new_attr_dict.items():
            real_word = idx2token[idx].replace(",", "").replace("?", "").replace(".", "").replace("!", "").replace("?", "").replace(";", "").replace('"', "").strip()
            word2attributions[real_word].append(score)
            pred_token_dic[real_word].append(prob)

        counter += 1

    return word2attributions, pred_token_dic, [dotdict(x) for x in new_data_record]


def __main__():
    exit()