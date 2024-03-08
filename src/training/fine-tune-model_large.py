from transformers import AutoTokenizer
import pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, ElectraForSequenceClassification, AdamW, GPT2ForSequenceClassification, GPT2Config #BertConfig, 
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch
import numpy as np
import time
import datetime
import random
import yaml
import sys
import os
import wandb
from apex import amp ##### DYNAMIC LOSS SCALING
from ..utils.model_utils import *

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

"""
https://arxiv.org/pdf/1912.10165.pdf :
To train our model we follow a procedure largely based on the training procedures described in
Radford et al. (2019) with a few differences. All training is performed with a maximum sequence
length of 512 tokens. In the full dataset training setting we utilize a learning rate of 4 × 10−5
and
a batch size of 128. When training with a quarter of the dataset we then used a learning rate of
3 × 10−5
and a batch size of 32. Our learning rate has a warmup period over 1% of the total training
iterations before decaying according to a single cycle cosine decay schedule over 10 epochs. We
utilize an Adam optimizer (Kingma and Ba, 2014) with decoupled weight decay (Loshchilov and
Hutter, 2019) λ = 0.01. All our models are trained efficiently on V100 GPUs by utilizing mixed
precision training with dynamic loss scaling (Micikevicius et al., 2017). Additionally, we use global
gradient norm clipping of 1.0 to improve the stability of training large models. Lastly, we utilize
attention and hidden state dropout (Srivastava et al., 2014) values of 0.1.

"""

"""
Save pre-trained model locally and load at start of code
Save data locally and load at start of code
"""

### Sanity check we have access to CUDA
print(torch.version.cuda)
print(torch.cuda.is_available()) # True)


MODEL = sys.argv[1]
DATA = sys.argv[2]


    
##################################################################################### KEEP THESE HYPERPARAMETERS STANDARD #################################################################################################

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


###########################################################################################################################################################################################################################

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project=f"{DATA}-{MODEL}-Finetuning",
    name=f"Raytuned",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-5,
    "batch_size": best_config['batch_size'],
    "architecture": MODEL,
    "dataset": DATA,
    "epochs": best_config['epochs'],
    "random_seed": best_config['random_seed']
    }
) ### To sync online, run in terminal: wandb sync wandb/dryrun-folder-name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get model configuration.
print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=f"gpt2-medium", num_labels=num_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=f"gpt2-medium")
# default to left padding
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token


# Get the actual model.
print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=f"gpt2-medium", config=model_config)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)
  
gpt2classificationcollator = ClassificationCollator(tokenizer=tokenizer,
                                                    text_label = text_label,
                                                    idx_label = "index",
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
                              batch_size = best_config['batch_size'], # Trains with this batch size,
                              collate_fn=gpt2classificationcollator)

print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
val_dataloader = DataLoader(
            dev_dataset,  # The training samples.
            sampler = RandomSampler(dev_dataset), # Select batches randomly
            batch_size = best_config['batch_size'], # Validates with this batch size.
            collate_fn=gpt2classificationcollator)
        


# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
test_dataloader = DataLoader(
            test_dataset,  # The training samples.
            sampler = RandomSampler(test_dataset), # Select batches randomly
            batch_size = 1, # Test with this batch size.
            collate_fn=gpt2classificationcollator)
        


print('Created `eval_dataloader` with %d batches!'%len(val_dataloader))

    

# Get the actual model.
print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=f"gpt2-medium", config=model_config)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

###############################################################################################

print("Freezing following layers")

# ct=0
# for child in model.children():
#     ct += 1
#     print(ct)
#     if ct > 1 : ### Only update linear layer params
#         for param in child.parameters():
#             param.requires_grad = False
            
#         print("Training:")
#         print(child)

### Go through heads
for i, m in enumerate(model.transformer.h):
    if i < 9:
        print(m)
        for param in m.parameters():
            param.requires_grad = False # Freeze first 9 heads
    
# for parameter in model.transformer.ln_f.parameters():        
#     parameter.requires_grad = True

# for parameter in model.lm_head.parameters():        
#     parameter.requires_grad = True
        
###########################################################################################
    
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr = best_config['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5 ###4e-5 from Puri & Catanzero
                  eps = best_config['adam_eps'], # args.adam_epsilon  - default is 1e-8.
                  weight_decay = best_config['weight_decay'] ### Puti & Cantazaro
                )


model, optimizer = amp.initialize(model, optimizer, opt_level="O1") ###O1 allows dynamic lsos scaling. See https://blog.paperspace.com/automatic-mixed-precision-using-pytorch/

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * best_config['epochs']

# Create the learning rate scheduler. # Puri & Catanzero
scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = int(total_steps*0.01), # Our learning rate has a warmup period over 1% of the total training iterations before decaying
                                            num_training_steps = total_steps)


"""Our learning rate has a warmup period over 1% of the total training
iterations before decaying according to a single cycle cosine decay schedule over 10 epochs. We
utilize an Adam optimizer (Kingma and Ba, 2014) with decoupled weight decay (Loshchilov and
Hutter, 2019) λ = 0.01"""


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
    
    #### ONLY UPDATE CLASSIFICATION LAYER
    
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
        b_input_ids = batch['input_ids'].to(device) #### IN OTHER ONES I USE TORCH.STACK AROUND BATCH['INPUT_IDS']
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
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels).loss
        
        
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # # Perform a backward pass to calculate the gradients.
        # loss.backward()
        
        ### FOR DYNAMIC LOSS SCALING
        with amp.scale_loss(loss, optimizer) as scaled_loss: 
            scaled_loss.backward()

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
                                   token_type_ids=None, 
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
model_name = f"Models/{DATA}/{MODEL_BASE}-{datetime.date.today()}.pt"
print(f"Model saved to {model_name}")
torch.save(model.state_dict(), model_name)
model.config.to_json_file(f"Models/{DATA}/gpt2-config.json")


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
                                token_type_ids=None, 
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

