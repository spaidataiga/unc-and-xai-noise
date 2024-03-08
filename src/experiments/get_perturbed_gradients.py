"""
This is source code to collect all the desired saliency maps for a model output, regardless of the model.

    ##### Go through 3 noise patterns: random, human, gradient
    ##### Go through the 4 kidns of noise: unk, mask, synonym, charswap
    ##### for unk and mask add this before tokenization:
    ##### At dataloader level, select the feature we will use for input, if token type noise, 
    
    # g = { 'TOKEN': tokenizer.unk_token }

    # print(out['random_25'].format(**g))
    THEN map tokenize to text!

"""

import torch
from captum.attr._utils.visualization import *
import numpy as np
from ..utils.gradient_utils import *
from ..utils.general_utils import *
from ..utils.model_utils import *
import sys

pts = ['S', 'R']
nks = ['token', 'charswap', 'synonym', 'butterfingers', 'wordswap', 'charinsert', 'l33t']
lvls = ['05','10','25', '50', '70', '80', '90', '95']

if __name__ == "__main__":
    ######################################## DECLARE DATA TO USE AND MODEL TYPE ##############################################
    MODEL = sys.argv[1].lower()
    DATA = sys.argv[2]
    PATTERN = sys.argv[3].lower()

    MODEL_BASE, text_label, is_lower, MAX_LENGTH, idx_label, num_labels = get_model_details(MODEL,DATA)

    assert PATTERN in ['random', 'human', 'gradient']
    print(f"Valid pattern type: {PATTERN}")
    print()

        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = get_model(MODEL_BASE, DATA, device)

    #### Wrapper with custom forward function required for Captum
    modelwrapper = ModelWrapper(model, device)
    modelwrapper.to(device)
    modelwrapper.eval()
    modelwrapper.zero_grad()
        
    #############################################################################################################################

    if PATTERN == 'human':
        for pt in pts: # strategic or random
            for nk in nks: # kinds of noise
                for lvl in lvls: # levels of noise
                    data = f"{PATTERN}-{pt}_{nk}_{lvl}"
                    if nk=='token':
                        for tt in ['unk', 'mask']: #insert unknown or mask token
                            DOWNSTREAM_FOLDER = f"{PATTERN}/{pt}/{nk}-{tt}/{lvl}/"
                            print("Preparing data for: ", DOWNSTREAM_FOLDER)
                            dataloader = get_data(DATA, tokenizer, data, token=tt)
                            run_and_write_gradients(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                            print()
                    else:
                        DOWNSTREAM_FOLDER = f"{PATTERN}/{pt}/{nk}/{lvl}/"
                        print("Preparing data for: ", DOWNSTREAM_FOLDER)
                        dataloader = get_data(DATA, tokenizer, data)
                        run_and_write_gradients(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                        print()
                        
        pt = 'A' ### Now check when we only perturb all human annotations
        for nk in nks:
            data = f"{PATTERN}-{pt}_{nk}"
            if nk=='token':
                for tt in ['unk', 'mask']: #insert unknown or mask token
                    DOWNSTREAM_FOLDER = f"{PATTERN}/{pt}/{nk}-{tt}/"
                    print("Preparing data for: ", DOWNSTREAM_FOLDER)
                    dataloader = get_data(DATA, tokenizer, data, token=tt)
                    run_and_write_gradients(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                    print()
            else:
                DOWNSTREAM_FOLDER = f"{PATTERN}/{pt}/{nk}/"
                print("Preparing data for: ", DOWNSTREAM_FOLDER)
                dataloader = get_data(DATA, tokenizer, data)
                run_and_write_gradients(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                print()
                        
    elif PATTERN == 'gradient':
        for nk in nks: # kinds of noise
            for lvl in lvls: # levels of noise
                data = f"{PATTERN}-{MODEL}_{nk}_{lvl}"
                if nk=='token':
                    for tt in ['unk', 'mask']: #insert unknown or mask token
                        DOWNSTREAM_FOLDER = f"{PATTERN}/{nk}-{tt}/{lvl}/"
                        print("Preparing data for: ", DOWNSTREAM_FOLDER)
                        dataloader = get_data(DATA, tokenizer, data, token=tt)
                        run_and_write_gradients(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                        print()
                else:
                    DOWNSTREAM_FOLDER = f"{PATTERN}/{nk}/{lvl}/"
                    print("Preparing data for: ", DOWNSTREAM_FOLDER)
                    dataloader = get_data(DATA, tokenizer, data)
                    run_and_write_gradients(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                    print()
                    
    else: # random
        for nk in nks: # kinds of noise
            for lvl in lvls: # levels of noise
                data = f"{PATTERN}_{nk}_{lvl}"
                if nk=='token':
                    for tt in ['unk', 'mask']: #insert unknown or mask token
                        DOWNSTREAM_FOLDER = f"{PATTERN}/{nk}-{tt}/{lvl}/"
                        print("Preparing data for: ", DOWNSTREAM_FOLDER)
                        dataloader = get_data(DATA, tokenizer, data, token=tt)
                        run_and_write_gradients(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                        print()
                else:
                    DOWNSTREAM_FOLDER = f"{PATTERN}/{nk}/{lvl}/"
                    print("Preparing data for: ", DOWNSTREAM_FOLDER)
                    dataloader = get_data(DATA, tokenizer, data)
                    run_and_write_gradients(modelwrapper, dataloader, tokenizer, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                    print()
                    
    print("Completed.")