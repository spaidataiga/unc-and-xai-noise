from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, ElectraForSequenceClassification, AdamW, GPT2ForSequenceClassification, GPT2Config #Trainer, TrainingArguments  #BertConfig, 
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
import torch
import numpy as np
import time
import datetime
import os
import wandb
import sys
from apex import amp ##### DYNAMIC LOSS SCALING
from ray import train,tune
from ..utils.model_utils import *
# from ray.train.huggingface import TransformersTrainer
# from ray.train import ScalingConfig
 # from ray.train import Trainer
# import ray.tune.integration.torch.DistributedTrainableCreator as DTC
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
import logging

## Following https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html
### https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
## https://github.com/ray-project/ray/issues/10986
## https://github.com/ray-project/ray/issues/10910
      
MODEL = sys.argv[1]
DATA = sys.argv[2]

def model_init(data):
    global MODEL
    global tokenizer
    global MAX_LENGTH
    global text_label
    
    MODEL_BASE, text_label, is_lower, MAX_LENGTH, idx_label, num_labels = get_model_details(MODEL,DATA)
    
    if MODEL == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(f'facebook/opt-350m', ## Saved locally
            do_lower_case = is_lower)
        
        model = OPTForSequenceClassification.from_pretrained(
            "facebook/opt-350m", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_labels, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            # torch_dtype=torch.float16, attn_implementation="flash_attention_2" ### Speeds up inference
        )

    elif MODEL == 'gpt2-medium': 
        # Get model's tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=f"{MODEL}", do_lower_case = is_lower)
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=f"{MODEL}", num_labels=num_labels)


        model = GPT2ForSequenceClassification.from_pretrained(
            f'{MODEL}', # Use the 12-layer BERT model, with an uncased vocab.
            config = model_config
        )
        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id

        ### Go through heads
        for i, m in enumerate(model.transformer.h):
            if i < 9:
                print(m)
                for param in m.parameters():
                    param.requires_grad = False # Freeze first 9 heads
                    
    else:
        print("Train smaller models using the other method")
        return
            
    return model
    
#### Inner function to map tokenizer
def tokenize_function(examples):
    global tokenizer
    global MAX_LENGTH
    global text_label
    if type(text_label) == list: ##esnli, pair task
        return tokenizer(examples[text_label[0]],examples[text_label[1]], padding="max_length", max_length=MAX_LENGTH, truncation=True) ## Max length of dataset is 52 tokens. Rounded to nearest multiple of 8
    else:
        return tokenizer(examples[text_label], padding="max_length", max_length=MAX_LENGTH, truncation=True) ## Max length of dataset is 52 tokens. Rounded to nearest multiple of 8

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(logits, labels):
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train_model(config):
        
    global tokenizer
    global MAX_LENGTH
    global text_label
    global idx_label
    global is_lower

    
    model = model_init(config['data'])

    data_path = f"/home/fvd442/project/noise-paper/Data/{config['data']}"
    
    
    full_dataset = load_from_disk(data_path)#.map(tokenize_function, batched=True)

    train_dataset = full_dataset['train']
    dev_dataset = full_dataset['validation']
    #test_dataset =  full_dataset['test']
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # model = train.torch.prepare_model(model)
    
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                    lr = config['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5 ###4e-5 from Puri & Catanzero
                    eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                    weight_decay = 0.01 ### Puti & Cantazaro
                    )
    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1") ###O1 allows dynamic lsos scaling. See https://blog.paperspace.com/automatic-mixed-precision-using-pytorch/

    
    gpt2classificationcollator = ClassificationCollator(tokenizer=tokenizer,
                                                        text_label = text_label,
                                                        idx_label = "index",
                                                        max_seq_len=MAX_LENGTH)
    
    # if torch.cuda.is_available():
    #     # model = nn.DistributedDataParallel(model).cuda()
    #     model = DDP(model) #.cuda()
    
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                                train_dataset,  # The training samples.
                                sampler = RandomSampler(train_dataset), # Select batches randomly ### TRY Shuffle=true?
                                batch_size = config['batch_size'], # Trains with this batch size,
                                collate_fn=gpt2classificationcollator)

    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))
    
    # train_dataloader = train.torch.prepare_data_loader(train_dataloader)

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    val_dataloader = DataLoader(
                dev_dataset,  # The training samples.
                sampler = RandomSampler(dev_dataset), # Select batches randomly
                batch_size = config['batch_size'], # Validates with this batch size.
                collate_fn=gpt2classificationcollator)
    
    # val_dataloader = train.torch.prepare_data_loader(val_dataloader)
    
# To restore a checkpoint, use `train.get_checkpoint()`.
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * config['epochs']

    # Create the learning rate scheduler. # Puri & Catanzero
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(total_steps*0.01), # Our learning rate has a warmup period over 1% of the total training iterations before decaying
                                                num_training_steps = total_steps)
    
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    
    
    # trainer = Trainer(backend="torch", num_workers=2, use_gpu=True)


    # For each epoch...
    for epoch_i in range(0, config['epochs']):
        
        # ========================================
        #               Training 
        # ========================================
        
        #### ONLY UPDATE CLASSIFICATION LAYER
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, config['epochs']))
        print('Training...')
        
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for batch in train_dataloader:
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
            
            # if MODEL == 'opt':
            #     loss = model(b_input_ids, 
            #                 attention_mask=b_input_mask, 
            #                 labels=b_labels).loss
            # else:
            #     loss = model(b_input_ids, 
            #                 token_type_ids=None, 
            #                 attention_mask=b_input_mask, 
            #                 labels=b_labels).loss
            loss = model(b_input_ids, 
                        # token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels).loss
            
            # logging.warning("LOSS:",loss)
            
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # # Perform a backward pass to calculate the gradients.
            # loss.backward()
            
            #loss.sum().backward()
            
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
            
        print("")
        print("Running Validation...")

    
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0

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
                                    # token_type_ids=None, 
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

        # log metrics to wandb
        # wandb.log({"acc": avg_val_accuracy, "loss": avg_train_loss})
        
        os.makedirs(f"{MODEL}-raytune", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), f"{MODEL}-raytune/checkpoint.pt")
        checkpoint = Checkpoint.from_directory(f"{MODEL}-raytune")
        train.report({"loss": (avg_val_loss), "accuracy": avg_val_accuracy}, checkpoint=checkpoint)
        print("Finished Training")
        

def main(data, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    
    config = {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "epochs": tune.choice(range(1, 5)),
        "batch_size": tune.choice([4, 8, 16, 32]), #64, 128
        "data": data
    }
    
    
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 4, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")
    
    #trainer = Trainer(backend="torch", num_workers=2, use_gpu=True)
    #trainable = trainer.to_tune_trainable(train_model)

    # trainable= DTC(train_model, num_workers=1, num_cpus_per_worker=4, num_gpus_per_worker=2, backend= 'nccl')
    
    # scaling_config = ScalingConfig(num_workers=4, use_gpu=True)
    # ray_trainer = TransformersTrainer(
    #     trainer_init_per_worker=1,
    #     scaling_config=scaling_config,
    #     datasets={"train": ray_train_ds, "evaluation": ray_eval_ds},
    # )
    # result = ray_trainer.fit()


    # results = tune.run(trainable, num_samples = 10, scheduler=scheduler, config=config)
    # best_result = results.get_best_config("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

### Sanity check we have access to CUDA
print(torch.version.cuda)
print(torch.cuda.is_available()) # True)

assert DATA in ['SemEval', 'SST-2', 'hatexplain', 'esnli']
print("Valid data inputted: ", DATA)

main(DATA, num_samples=25, max_num_epochs=5, gpus_per_trial=2) 
