from transformers import AutoTokenizer
import pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, ElectraForSequenceClassification, AdamW, GPT2ForSequenceClassification, GPT2Config #BertConfig, 
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
import numpy as np
import time
import datetime
import random
import sys
import os
import wandb
from ..utils.model_utils import *
import yaml

"""
Save pre-trained model locally and load at start of code
Save data locally and load at start of code
"""

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


### Sanity check we have access to CUDA
print(torch.version.cuda)
print(torch.cuda.is_available()) # True)

# os.environ['CURL_CA_BUNDLE'] = '' #### Should help us avoid SSL Error according to https://stackoverflow.com/questions/75110981/sslerror-httpsconnectionpoolhost-huggingface-co-port-443-max-retries-exce
### os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
#os.environ["WANDB_MODE"] = "dryrun"


################## TAKE MODEL AND DATA AS ARGUEMENTS ##############################
MODEL = sys.argv[1].lower()
DATA = sys.argv[2]


##################################################################################### KEEP THESE HYPERPARAMETERS STANDARD #################################################################################################

MODEL_BASE, text_label, is_lower, MAX_LENGTH, idx_label, num_labels = get_model_details(MODEL,DATA)

with open("config.yaml", "r") as yamlfile:
    model_configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Acquired model parameters")

best_config = model_configs[MODEL][DATA.lower()]

random.seed(best_config['random_seed'])
np.random.seed(best_config['random_seed'])
torch.manual_seed(best_config['random_seed'])
torch.cuda.manual_seed_all(best_config['random_seed'])

###########################################################################################################################################################################################################################

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project=f"{DATA}-{MODEL}-Finetuning",
    name=f"{DATA}_best_{MODEL_BASE}",
    # track hyperparameters and run metadata
    config= best_config
) ### To sync online, run in terminal: wandb sync wandb/dryrun-folder-name

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


#########################################################################################################################################################################################

##### Get datasets and tokenize them
full_dataset = load_from_disk(f"./Data/{DATA}") #.map(tokenize_function, batched=True)

train_dataset = full_dataset['train']
dev_dataset = full_dataset['validation']
if DATA == "SST-2":
    test_dataset = load_from_disk(f"./Data/Hummingbird") #.map(tokenize_function, batched=True)['test'] #full_dataset['test']
else:
    test_dataset = full_dataset['test']

print("Training size: ", train_dataset.num_rows)

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly ### TRY Shuffle=true?
            batch_size = best_config['batch_size'], # Trains with this batch size.
            collate_fn=collator
        )

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
val_dataloader = DataLoader(
            dev_dataset,  # The training samples.
            sampler = RandomSampler(dev_dataset), # Select batches randomly
            batch_size = best_config['batch_size'], # Validates with this batch size.
            collate_fn=collator
        )


# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
test_dataloader = DataLoader(
            test_dataset,  # The training samples.
            sampler = RandomSampler(test_dataset), # Select batches randomly
            batch_size = 1, # Test with this batch size.
            collate_fn=collator
        )




# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 

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
elif MODEL.lower() == "opt":
    model = OPTForSequenceClassification.from_pretrained(
    "facebook/opt-350m", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = num_labels, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
    #torch_dtype=torch.float16, attn_implementation="flash_attention_2" ### Speeds up inference
)
else: #model GPT2
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=f"gpt2-medium", num_labels=num_labels)
    model = GPT2ForSequenceClassification.from_pretrained(
        MODEL_BASE, # Use the 12-layer BERT model, with an uncased vocab.
        config = model_config
    )
    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id
    
# Tell pytorch to run this model on the GPU.
model.to(device) ###################################################################################################### !!!!!!!!!!!!!!!!!!!!!!!!!!
print('Model loaded to `%s`'%device)

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

if best_config['llrd']:
    grouped_optimizer_params = get_optimizer_grouped_parameters(
    model, MODEL.lower(), 
    best_config['learning_rate'], best_config['weight_decay'], 
    best_config['llrd']
)
    
    optimizer = AdamW(
        grouped_optimizer_params,
        lr= best_config['learning_rate'],
        eps = best_config['adam_eps'],
        betas=(best_config['adam_b1'], best_config['adam_b2'])
    )

else:  
    optimizer = AdamW(model.parameters(),
                        lr = best_config['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = best_config['adam_eps'],
                        betas=(best_config['adam_b1'], best_config['adam_b2']),
                        weight_decay=best_config['weight_decay']
                    )


# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * best_config['epochs']

# Create the learning rate scheduler.

assert best_config['decay_type'] in ['linear', 'cosine']

if best_config['decay_type'] == 'linear':
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(best_config['warmup_frac'] * total_steps), # Default value in run_glue.py
                                                num_training_steps = total_steps)
if best_config['decay_type'] == 'cosine':
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(best_config['warmup_frac'] * total_steps), # Our learning rate has a warmup period over 1% of the total training iterations before decaying
                                                num_training_steps = total_steps)


# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, best_config['epochs']):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, best_config['epochs']))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss = model(b_input_ids, 
                    # token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels).loss

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in val_dataloader: 
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            outputs      = model(b_input_ids, 
                                #    token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += outputs.loss.item()
        # Move logits and labels to CPU
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(val_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    
    # log metrics to wandb
    wandb.log({"acc": avg_val_accuracy, "loss": avg_train_loss})

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# ========================================
#               SAVE MODEL
# ========================================
model_name = f"Models/{DATA}/{MODEL.lower()}.pt" #{MODEL_BASE}-{best_config['epochs']}epochs-{best_config['random_seed']}_{datetime.date.today()}.pt"
print(f"Model saved to {model_name}")
torch.save(model.state_dict(), model_name)
model.config.to_json_file(f"Models/{DATA}/{MODEL.lower()}-config.json")


# ========================================
#               Final results on test
# ========================================
# After saving the model, let's see how it performed on the test dataset

print("")
print("Testing...")

t0 = time.time()

# Put the model in evaluation mode--the dropout layers behave differently
# during evaluation.
model.eval()

# Tracking variables 
total_test_accuracy = 0

# Evaluate data for one epoch
for batch in test_dataloader: 
    
    # Unpack this training batch from our dataloader. 
    #
    # As we unpack the batch, we'll also copy each tensor to the GPU using 
    # the `to` method.
    #
    # `batch` contains three pytorch tensors:
    #   [0]: input ids 
    #   [1]: attention masks
    #   [2]: labels 
    b_input_ids = batch['input_ids'].to(device)
    b_input_mask = batch['attention_mask'].to(device)
    b_labels = batch['labels'].to(device)
    
    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():        

        # Forward pass, calculate logit predictions.
        # token_type_ids is the same as the "segment ids", which 
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        outputs = model(b_input_ids, 
                                # token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)

    # Move logits and labels to CPU
    logits = outputs.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    total_test_accuracy += flat_accuracy(logits, label_ids)
    

# Report the final accuracy for this validation run.
avg_test_accuracy = total_test_accuracy / len(test_dataloader)
print("Test Accuracy: {0:.2f}".format(avg_test_accuracy))