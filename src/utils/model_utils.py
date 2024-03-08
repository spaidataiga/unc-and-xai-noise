import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, ElectraForSequenceClassification, AdamW, GPT2ForSequenceClassification, GPT2Config, BertConfig, ElectraConfig, RobertaConfig, OPTForSequenceClassification, OPTConfig
import torch.nn as nn
import numpy as np
from datasets import load_from_disk, Dataset
import datetime
import sys
import torch


def get_model_details(MODEL, DATA):
    
    with open('/home/fvd442/project/noise-paper/Resources/models.txt') as f:
        models = [line.strip() for line in f.readlines()]
        
    with open('/home/fvd442/project/noise-paper/Resources/datasets.txt') as f:
        datasets = [line.strip() for line in f.readlines()]
    
    assert DATA in datasets
    print("Valid data inputted: ", DATA)

    assert MODEL in models
    print("Valid model inputted: ", MODEL)

    if not 'gpt2' in MODEL and MODEL != 'opt':
        MODEL_BASE = MODEL.lower() + "-base"
    else:
        MODEL_BASE = MODEL
        

    if DATA == "SemEval":
        if MODEL.lower() == 'bert':
            MODEL_BASE = MODEL_BASE + "-cased" ##### TESTING OUT BERT UNCASED
        is_lower = False
        MAX_LENGTH = 256
        text_label = "text"
        idx_label = "index"
        num_labels = 2
    elif DATA == "hatexplain":
        if MODEL.lower() == 'bert':
            MODEL_BASE = MODEL_BASE + "-uncased" ##### TESTING OUT BERT UNCASED
        is_lower = True
        MAX_LENGTH = 256
        text_label = "text"
        idx_label = "index"
        num_labels = 3
    elif DATA == "esnli":
        if MODEL.lower() == 'bert':
            MODEL_BASE = MODEL_BASE + "-cased" ##### TESTING OUT BERT UNCASED
        is_lower = False
        MAX_LENGTH = 256
        text_label = ["text_1", "text_2"]
        idx_label = "index"
        num_labels = 3
    else: ###SST-2
        if MODEL.lower() == 'bert':
            MODEL_BASE = MODEL_BASE + "-uncased" ### To match https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
        is_lower = True
        text_label = "sentence"
        MAX_LENGTH = 128
        idx_label = "idx" 
        num_labels = 2
        
    return MODEL_BASE, text_label, is_lower, MAX_LENGTH, idx_label, num_labels


class ModelWrapper(nn.Module): ##### Wrapper from https://github.com/mt-upc/transformer-contributions/blob/main/src/contributions.py

    def __init__(self, model, device):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.device = device
    
    def compute_pooled_outputs(self, embedding_output, attention_mask=None, head_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

        encoder_outputs = self.model(inputs_embeds=embedding_output, attention_mask = attention_mask)
        return encoder_outputs[0]

    def get_prediction(self, input_model):
        output = self.model(input_model, output_hidden_states=True, output_attentions=True)
        logits = output['logits']
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_class_ind = torch.argmax(probs).detach().cpu().item()
        pred = torch.max(probs)#[pred_ind]
        prob_pred_class = probs[0][pred_class_ind].detach().cpu().item()
        
        return pred_class_ind, prob_pred_class

    def forward(self, input_embedding):      
        logits = self.compute_pooled_outputs(input_embedding)

        return torch.softmax(logits, dim=-1)

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
    def get_embedding(self, input_ids):
        if self.model.config.model_type == 'bert':
            input_embedding = self.model.bert.embeddings(input_ids)
        elif self.model.config.model_type == 'electra':
            input_embedding = self.model.electra.embeddings(input_ids)
        elif self.model.config.model_type == 'roberta':
            input_embedding = self.model.roberta.embeddings(input_ids)
        elif self.model.model.decoder.config.model_type == 'opt':
            input_embedding = self.model.model.decoder.embed_tokens(input_ids)
        else: #gpt2
            # input_embedding = self.model.transformer.wte(input_ids)
            input_embedding = self.model.transformer.wte(input_ids) + self.model.transformer.wpe(torch.arange(0, input_ids.size(dim=1)).to(self.device).unsqueeze(0))

        return input_embedding
    
    
def get_optimizer_grouped_parameters(
    model, model_type, 
    learning_rate, weight_decay, 
    layerwise_learning_rate_decay
): # From this notebook:  https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning/notebook
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters
    
    
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class ClassificationCollator(object):
    def __init__(self, tokenizer, text_label, idx_label, token_replacement=None, test_only=False, max_seq_len=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if type(text_label) == list:
            self.text_labels = text_label
            self.pair_task = True
        else:
            self.pair_task = False
            self.text_labels = text_label
        self.text_labels = text_label
        self.idx_label = idx_label
        if test_only:
            self.idx_label = 'index'
            
        self.ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
        self.sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
        self.cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
        self.eos_token_id = tokenizer.eos_token_id # Used by GPT2 for marking end of sequence
    
        if token_replacement:
            self.fill_tokens = True
            if token_replacement == 'unk':
                self.replacement = { 'TOKEN': tokenizer.unk_token }
            elif token_replacement == 'mask':
                self.replacement = { 'TOKEN': tokenizer.mask_token }
        else:
            self.fill_tokens = False
                    
        return
    
    def __call__(self, sequences):
        if self.pair_task:
            if self.fill_tokens:
                # text1 = [sequence[self.text_labels[0]].format(**self.replacement) for sequence in sequences]
                # text2 = [sequence[self.text_labels[1]].format(**self.replacement) for sequence in sequences]
                
                texts = [sequence[self.text_labels[0]].format(**self.replacement) + '|' + sequence[self.text_labels[1]].format(**self.replacement) for sequence in sequences]
            else:
                # text1 = [sequence[self.text_labels[0]] for sequence in sequences]
                # text2 = [sequence[self.text_labels[1]] for sequence in sequences]
                texts = [sequence[self.text_labels[0]] + '|' + sequence[self.text_labels[1]] for sequence in sequences]
                
            inputs = self.tokenizer(texts, #text1, text2,
                            return_tensors='pt',
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_seq_len)
        else:
            if self.fill_tokens:
                texts = [sequence[self.text_labels].format(**self.replacement) for sequence in sequences]
            else:
                texts = [sequence[self.text_labels] for sequence in sequences]
                
            inputs = self.tokenizer(texts,
                        return_tensors='pt',
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_seq_len)

        labels = [int(sequence['label']) for sequence in sequences]
        ids = [int(sequence[self.idx_label]) for sequence in sequences]

        inputs.update({'labels': torch.tensor(labels)})
        inputs.update({'index': torch.tensor(ids)})
        inputs.update({'ref_ids': torch.stack([torch.tensor([val if val in [self.cls_token_id, self.sep_token_id, self.eos_token_id] else self.ref_token_id for val in inp]) for inp in inputs['input_ids']])})
            
        return inputs
    
    


def get_model_data(model_name, data_name, device, max_size=None, sequential=False):
    
    ####################### Determine Tokenizer rules #############################
    is_sfx = False
    
    if "_" in data_name:
        is_sfx=True
        sfx = data_name.split('_')[1]
        data_name = data_name.split('_')[0]
        
    idx_label = "index"
    text_label = "text"
    
        
    if data_name == 'SST-2':
        MAX_LENGTH = 128
        is_lower = True
        num_labels = 2
    elif "hatexplain" in data_name:
        is_lower = True
        MAX_LENGTH = 256
        num_labels = 3
    elif data_name == "esnli":
        is_lower = False
        MAX_LENGTH = 256
        num_labels = 3
        text_label = ["text_1", "text_2"]
    else: ## SemEval
        MAX_LENGTH = 256
        is_lower = False
        num_labels = 2
        
    
    if is_sfx:
        test_data = data_name + "_" + sfx
    else:
        test_data = data_name

        
    if "electra" in model_name: ## Must be lowercase
        is_lower = True
        
    tokenizer = AutoTokenizer.from_pretrained(f'./Tokenizers/Pretrained/{model_name}.pt', ## Saved locally
                                                do_lower_case = is_lower ## Check if BERT is uncased
                                                )
    
    if 'gpt2' in model_name:
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
    
    collator = ClassificationCollator(tokenizer=tokenizer,
                                      text_label=text_label,
                                      idx_label=idx_label,
                                      test_only=True,
                                      max_seq_len=MAX_LENGTH)
    

    ##### Get datasets and tokenize them
    if max_size:
        test_dataset = load_from_disk(f"./Data/Clean/{test_data}")['test'][:max_size]
        test_dataset = Dataset.from_dict(test_dataset)
    else:
        test_dataset = load_from_disk(f"./Data/Clean/{test_data}")['test']

    print("Dataset size: ", test_dataset.num_rows)

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    test_dataloader = DataLoader(
                test_dataset,  # The training samples.
                sampler = SequentialSampler(test_dataset) if sequential else RandomSampler(test_dataset), # Select batches randomly
                batch_size = 1, # Test with this batch size.
                collate_fn=collator
            )
    
    print("Loaded data: ", data_name)

    ########################### LOAD MODEL ####################################################

    if model_name.split('-')[0] == 'bert':
        
        model_config = BertConfig.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/bert-config.json', num_labels=num_labels)
        model = BertForSequenceClassification(
            model_config
        )
        model.load_state_dict(torch.load(f'./Models/{data_name}/bert.pt', map_location=device))
    elif model_name == "opt":
        model_config = OPTConfig.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/opt-config.json', num_labels=num_labels)
        model = OPTForSequenceClassification(
            model_config,
            # torch_dtype=torch.float16, attn_implementation="flash_attention_2" ### Speeds up inference

        )
        model.load_state_dict(torch.load(f'./Models/{data_name}/opt.pt', map_location=device))
    elif model_name.split('-')[0] == 'roberta':
        
        model_config = RobertaConfig.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/roberta-config.json', num_labels=num_labels)
        model = RobertaForSequenceClassification(model_config)
        model.load_state_dict(torch.load(f'./Models/{data_name}/roberta.pt', map_location=device))    
    elif model_name.split('-')[0] == 'electra':
        model_config = ElectraConfig.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/electra-config.json', num_labels=num_labels)
        model = ElectraForSequenceClassification(model_config)
        model.load_state_dict(torch.load(f'./Models/{data_name}/electra.pt', map_location=device))    
        
    elif 'gpt2' in model_name.split('-')[0]:
        
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/gpt2-config.json', num_labels=num_labels)
        #model_config = GPT2Config(vocab_size=50258, num_labels=2)
        model = GPT2ForSequenceClassification.from_pretrained(
            f'./Models/{data_name}/gpt2-medium.pt', # Use the 12-layer BERT model, with an uncased vocab.
            config=model_config,
        )
        
        model.load_state_dict(torch.load(f'./Models/{data_name}/gpt2-medium.pt', map_location=device))    
        
        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id
        
    else:
        print("unaccepted model type. Received: ", model_name.split('-')[0])
        return -1
    
    model.to(device)
    
    print("Loaded model: ", model_name)
    return test_dataloader, model, tokenizer
    
def get_model(model_name, data_name, device):
    
    ####################### Determine Tokenizer rules #############################
    if "_" in data_name:
        is_sfx=True
        sfx = data_name.split('_')[1]
        data_name = data_name.split('_')[0]
        
    if data_name == 'SST-2':
        is_lower = True
        num_labels = 2
    elif "hatexplain" in data_name:
        is_lower = True
        num_labels = 3
    elif data_name == "esnli":
        is_lower = False
        num_labels = 3
    else: ## SemEval
        is_lower = False
        num_labels = 2
        

    if "electra" in model_name: ## Must be lowercase
        is_lower = True
        
        
    tokenizer = AutoTokenizer.from_pretrained(f'./Tokenizers/Pretrained/{model_name}.pt', ## Saved locally
                                                do_lower_case = is_lower ## Check if BERT is uncased
                                                )
    
    if 'gpt2' in model_name:
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
    

    ########################### LOAD MODEL ####################################################
    
    if model_name.split('-')[0] == 'bert':
        
        model_config = BertConfig.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/bert-config.json', num_labels=num_labels)
        model = BertForSequenceClassification(
            model_config
        )
        model.load_state_dict(torch.load(f'./Models/{data_name}/bert.pt', map_location=device))
    elif model_name.split('-')[0] == 'roberta':
        
        model_config = RobertaConfig.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/roberta-config.json', num_labels=num_labels)
        model = RobertaForSequenceClassification(model_config)
        model.load_state_dict(torch.load(f'./Models/{data_name}/roberta.pt', map_location=device))    
    elif model_name.split('-')[0] == 'electra':
        model_config = ElectraConfig.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/electra-config.json', num_labels=num_labels)
        model = ElectraForSequenceClassification(model_config)
        model.load_state_dict(torch.load(f'./Models/{data_name}/electra.pt', map_location=device))    
    elif model_name == 'opt':
        model_config = OPTConfig.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/opt-config.json', num_labels=num_labels)
        model = OPTForSequenceClassification(model_config)
        model.load_state_dict(torch.load(f'./Models/{data_name}/opt.pt', map_location=device))    
    elif 'gpt2' in model_name.split('-')[0]:
        
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=f'./Models/{data_name}/gpt2-config.json', num_labels=num_labels)
        model = GPT2ForSequenceClassification.from_pretrained(
            "gpt2-medium", # Use the 12-layer BERT model, with an uncased vocab.
            config = model_config
        )
        
        model.load_state_dict(torch.load(f'./Models/{data_name}/gpt2-medium.pt', map_location=device))    

        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id
        
    else:
        print("unaccepted model type. Received: ", model_name.split('-')[0])
        return -1
    
    model.to(device)
    
    print("Loaded model: ", model_name)
    return model, tokenizer


def get_data(data_name, tokenizer, text_label, token=None, use_extra = False, specific_index=False):
    
    if data_name == 'SST-2':
        MAX_LENGTH = 128
    elif "hatexplain" in data_name:
        MAX_LENGTH = 256
    elif data_name == "esnli":
        MAX_LENGTH = 256
    else: ## SemEval
        MAX_LENGTH = 256
    
    collator = ClassificationCollator(tokenizer=tokenizer,
                                      text_label=text_label,
                                      idx_label="index",
                                      token_replacement=token,
                                      test_only=True,
                                      max_seq_len=MAX_LENGTH)
    

    ##### Get datasets and tokenize them
    if use_extra:
        test_dataset = load_from_disk(f"./Data/ExtraNoise/{data_name}")
    else:
        test_dataset = load_from_disk(f"./Data/OPTNoise/{data_name}")
        
    if specific_index:
        for i, val in enumerate(test_dataset["index"]):
            if val == specific_index:
                # print(data_name, specific_index)
                break
        test_dataset = [test_dataset[i]]

    # print("Dataset size: ", test_dataset.num_rows)

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    test_dataloader = DataLoader(
                test_dataset,  # The training samples.
                sampler = SequentialSampler(test_dataset), # Select batches randomly
                batch_size = 1, # Test with this batch size.
                collate_fn=collator
            )
    
    print("Loaded data: ", data_name, text_label)
    
    return test_dataloader

def __main__():
    exit()