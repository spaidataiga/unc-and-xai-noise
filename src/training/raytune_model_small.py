from transformers import AutoTokenizer
import pandas as pd
from datasets import load_from_disk, load_metric
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, ElectraForSequenceClassification, AdamW, GPT2ForSequenceClassification, GPT2Config, Trainer, TrainingArguments, OPTForSequenceClassification #BertConfig, 
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DefaultDataCollator
import torch
import numpy as np
import time
import datetime
import random
import sys
import os
import wandb
from ..utils.model_utils import *
from ray import tune

"""
Save pre-trained model locally and load at start of code
Save data locally and load at start of code
"""

    
def my_hp_space_ray(trial):
    
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "num_train_epochs": tune.choice(range(1, 5)),
        "seed": tune.choice(range(1, 41)),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32]), #64, 128
    }
    
def tokenize_function(examples):
    if type(text_label) == list: ##esnli, pair task
        return tokenizer(examples[text_label[0]],examples[text_label[1]], padding="max_length", max_length=MAX_LENGTH, truncation=True) ## Max length of dataset is 52 tokens. Rounded to nearest multiple of 8
    else:
        return tokenizer(examples[text_label], padding="max_length", max_length=MAX_LENGTH, truncation=True) ## Max length of dataset is 52 tokens. Rounded to nearest multiple of 8

def flat_accuracy(eval_out):
    preds, labels = eval_out
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return {
        'accuracy': np.sum(pred_flat == labels_flat) / len(labels_flat),
    }

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def model_init():
    global MODEL
        
    if MODEL.lower() == 'bert':
        model = BertForSequenceClassification.from_pretrained(
            MODEL_BASE, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_labels, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    elif MODEL.lower() == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_BASE, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_labels, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    elif MODEL.lower() ==  'electra': # https://colab.research.google.com/github/elsanns/xai-nlp-notebooks/blob/master/electra_fine_tune_interpret_captum_ig.ipynb#scrollTo=_TioBrt5VhIr
        model = ElectraForSequenceClassification.from_pretrained(
            'google/' + MODEL_BASE + "-discriminator", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_labels, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    elif MODEL == "opt":
        model = OPTForSequenceClassification.from_pretrained(
        "facebook/opt-350m", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        torch_dtype=torch.float16, attn_implementation="flash_attention_2" ### Speeds up inference
    )
    else: #model GPT2
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=f"gpt2-medium", num_labels=2)
        model = GPT2ForSequenceClassification.from_pretrained(
            'gpt2-medium', # Use the 12-layer BERT model, with an uncased vocab.
            config = model_config
        )
        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id
        
    return model

if __name__ == '__main__':

    ### Sanity check we have access to CUDA
    print(torch.version.cuda)
    print(torch.cuda.is_available()) # True)

    # os.environ['CURL_CA_BUNDLE'] = '' #### Should help us avoid SSL Error according to https://stackoverflow.com/questions/75110981/sslerror-httpsconnectionpoolhost-huggingface-co-port-443-max-retries-exce
    ### os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'

    MODEL = sys.argv[1].lower()
    DATA = sys.argv[2]


    MODEL_BASE, text_label, is_lower, MAX_LENGTH, idx_label, num_labels = get_model_details(MODEL,DATA)

        

    ###########################################################################################################################################################################################################################

    # start a new wandb run to track this script
    wandb.login()
    os.environ['WANDB_PROJECT']=f'{MODEL}-{DATA}-Hypertuning'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(f'./Tokenizers/Pretrained/{MODEL_BASE}.pt', ## Saved locally
        do_lower_case = is_lower
        )

    if 'gpt2' in MODEL:
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        
        
    collator = ClassificationCollator(tokenizer=tokenizer,
                                        text_label=text_label,
                                        idx_label=idx_label,
                                        max_seq_len=MAX_LENGTH)

    # set_trace()


    #########################################################################################################################################################################################

    ##### Get datasets and tokenize them
    full_dataset = load_from_disk(f"./Data/{DATA}").map(tokenize_function, batched=True)

    train_dataset = full_dataset['train']
    dev_dataset = full_dataset['validation']
    test_dataset = full_dataset['test']


    # Evaluate during training and a bit more often
    # than the default to be able to prune bad trials early.
    # Disabling tqdm is a matter of preference.
    training_args = TrainingArguments(
        "test", evaluation_strategy="steps", eval_steps=5000,save_strategy="epoch", disable_tqdm=True)

    trainer = Trainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=full_dataset['train'],
        eval_dataset=full_dataset['validation'],
        model_init=model_init,
        compute_metrics=flat_accuracy,
        #data_collator=collator
    )

    # set_trace()

    # trainer.train()
    # Default objective is the sum of all metrics
    # when metrics are provided, so we have to maximize it.
    best_run = trainer.hyperparameter_search(
        direction="maximize", 
        backend="ray", 
        hp_space= my_hp_space_ray,
        n_trials=20, # number of trials
        local_dir = f'/home/fvd442/project/noise-paper/ray_tune/{MODEL}/'
    )

    print()
    print()
    print()
    print()
    print()
    print("*************************************************************************** MY OUTPUT HERE ***************************************************************************")
    print("BEST RUN RESULT: {}".format(best_run.objective))
    print("BEST RUN CONFIGURATION: ", best_run.hyperparameters)
    print("*********************************************************************************************************************************************************************")
    print()
    print()
    print()
    print()
    print()