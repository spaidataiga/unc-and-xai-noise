"""
This is source code to collect all the desired uncertainties for a model output, regardless of the model.
"""


import torch
from captum.attr._utils.visualization import *
import numpy as np
from ..utils.gradient_utils import *
from ..utils.general_utils import *
from ..utils.model_utils import *
import sys

overwrite= False
FWD_PASSES = 100
nks = ['token', 'charswap', 'synonym', 'butterfingers', 'wordswap', 'charinsert', 'l33t']
lvls = ['05','10','25', '50', '70', '80', '90', '95']

if __name__ == "__main__":
    ######################################## DECLARE DATA TO USE AND MODEL TYPE ##############################################
    MODEL = sys.argv[1].lower()
    DATA = sys.argv[2]
    PATTERN = sys.argv[3].lower()

    if len(sys.argv) > 4:
        pts = [sys.argv[4].upper()]
    else:
        pts = ['S', 'R']

    MODEL_BASE, text_label, is_lower, MAX_LENGTH, idx_label, num_labels = get_model_details(MODEL,DATA)

    assert PATTERN in ['random', 'human', 'gradient']
    print(f"Valid pattern type: {PATTERN}")
    print()

    if MODEL.lower() != 'gpt2-medium' and MODEL.lower() != 'opt':
        MODEL_BASE = MODEL.lower() + "-base"
    else:
        MODEL_BASE = MODEL
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = get_model(MODEL_BASE, DATA, device)

    #### Wrapper with custom forward function required for Captum
    modelwrapper = ModelWrapper(model, device)
    modelwrapper.to(device)
    modelwrapper.eval()
    modelwrapper.enable_dropout() ##### Turn on dropout for MC
    modelwrapper.zero_grad()
        
    #############################################################################################################################

    ##### Go through 3 noise patterns: random, human, gradient
    ##### Go through the 4 kidns of noise: unk, mask, synonym, charswap
    ##### for unk and mask add this before tokenization:
    ##### At dataloader level, select the feature we will use for input, if token type noise, 


    if PATTERN == 'human':
        for pt in pts: # strategic or random
            for nk in nks: # kinds of noise
                for lvl in lvls: # levels of noise
                    data = f"{PATTERN}-{pt}_{nk}_{lvl}"
                    if nk=='token':
                        for tt in ['unk', 'mask']: #insert unknown or mask token
                            DOWNSTREAM_FOLDER = f"{PATTERN}/{pt}/{nk}-{tt}/{lvl}/"
                            if not overwrite:
                                if os.path.isfile(f"../../Uncertainty/{MODEL}/{DOWNSTREAM_FOLDER}/{DATA}_uncertainty.tsv"):
                                    print(f"DATA EXISTS ALREADY FOR {DOWNSTREAM_FOLDER}/{MODEL}")
                                    continue
                            print("Preparing data for: ", DOWNSTREAM_FOLDER)
                            dataloader = get_data(DATA, tokenizer, data, token=tt)
                            run_and_write_uncertainty(modelwrapper, dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                            print()
                    else:
                        DOWNSTREAM_FOLDER = f"{PATTERN}/{pt}/{nk}/{lvl}/"
                        if not overwrite:
                            if os.path.isfile(f"../../Uncertainty/{MODEL}/{DOWNSTREAM_FOLDER}/{DATA}_uncertainty.tsv"):
                                print(f"DATA EXISTS ALREADY FOR {DOWNSTREAM_FOLDER}/{MODEL}")
                                continue
                        print("Preparing data for: ", DOWNSTREAM_FOLDER)
                        dataloader = get_data(DATA, tokenizer, data)
                        run_and_write_uncertainty(modelwrapper, dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                        print()
                        
            if pt == 'A': ### Now check when we only perturb all human annotations
                for nk in nks:
                    data = f"{PATTERN}-{pt}_{nk}"
                    if nk=='token':
                        for tt in ['unk', 'mask']: #insert unknown or mask token
                            DOWNSTREAM_FOLDER = f"{PATTERN}/{pt}/{nk}-{tt}/"
                            if not overwrite:
                                if os.path.isfile(f"../../Uncertainty/{MODEL}/{DOWNSTREAM_FOLDER}/{DATA}_uncertainty.tsv"):
                                    print(f"DATA EXISTS ALREADY FOR {DOWNSTREAM_FOLDER}/{MODEL}")
                                    continue
                            print("Preparing data for: ", DOWNSTREAM_FOLDER)
                            dataloader = get_data(DATA, tokenizer, data, token=tt)
                            run_and_write_uncertainty(modelwrapper, dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                            print()
                    else:
                        DOWNSTREAM_FOLDER = f"{PATTERN}/{pt}/{nk}/"
                        if not overwrite:
                            if os.path.isfile(f"../../Uncertainty/{MODEL}/{DOWNSTREAM_FOLDER}/{DATA}_uncertainty.tsv"):
                                print(f"DATA EXISTS ALREADY FOR {DOWNSTREAM_FOLDER}/{MODEL}")
                                continue
                        print("Preparing data for: ", DOWNSTREAM_FOLDER)
                        dataloader = get_data(DATA, tokenizer, data)
                        run_and_write_uncertainty(modelwrapper, dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                        print()
                        
    elif PATTERN == 'gradient':
        for nk in nks: # kinds of noise
            for lvl in lvls: # levels of noise
                data = f"{PATTERN}-{MODEL}_{nk}_{lvl}"
                if nk=='token':
                    for tt in ['unk', 'mask']: #insert unknown or mask token
                        DOWNSTREAM_FOLDER = f"{PATTERN}/{nk}-{tt}/{lvl}/"
                        if not overwrite:
                            if os.path.isfile(f"../../Uncertainty/{MODEL}/{DOWNSTREAM_FOLDER}/{DATA}_uncertainty.tsv"):
                                print(f"DATA EXISTS ALREADY FOR {DOWNSTREAM_FOLDER}/{MODEL}")
                                continue
                        print("Preparing data for: ", DOWNSTREAM_FOLDER)
                        dataloader = get_data(DATA, tokenizer, data, token=tt)
                        run_and_write_uncertainty(modelwrapper, dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                        print()
                else:
                    DOWNSTREAM_FOLDER = f"{PATTERN}/{nk}/{lvl}/"
                    if not overwrite:
                        if os.path.isfile(f"../../Uncertainty/{MODEL}/{DOWNSTREAM_FOLDER}/{DATA}_uncertainty.tsv"):
                            print(f"DATA EXISTS ALREADY FOR {DOWNSTREAM_FOLDER}/{MODEL}")
                            continue
                    print("Preparing data for: ", DOWNSTREAM_FOLDER)
                    dataloader = get_data(DATA, tokenizer, data)
                    run_and_write_uncertainty(modelwrapper, dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                    print()
                    
    else: # random
        for nk in nks: # kinds of noise
            for lvl in lvls: # levels of noise
                data = f"{PATTERN}_{nk}_{lvl}"
                if nk=='token':
                    for tt in ['unk', 'mask']: #insert unknown or mask token
                        DOWNSTREAM_FOLDER = f"{PATTERN}/{nk}-{tt}/{lvl}/"
                        if not overwrite:
                            if os.path.isfile(f"../../Uncertainty/{MODEL}/{DOWNSTREAM_FOLDER}/{DATA}_uncertainty.tsv"):
                                print(f"DATA EXISTS ALREADY FOR {DOWNSTREAM_FOLDER}/{MODEL}")
                                continue
                        print("Preparing data for: ", DOWNSTREAM_FOLDER)
                        dataloader = get_data(DATA, tokenizer, data, token=tt)
                        run_and_write_uncertainty(modelwrapper, dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL) # (modelwrapper, eval_dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL, FWD_PASSES=100):
                        print()
                else:
                    DOWNSTREAM_FOLDER = f"{PATTERN}/{nk}/{lvl}/"
                    if not overwrite:
                        if os.path.isfile(f"../../Uncertainty/{MODEL}/{DOWNSTREAM_FOLDER}/{DATA}_uncertainty.tsv"):
                            print(f"DATA EXISTS ALREADY FOR {DOWNSTREAM_FOLDER}/{MODEL}")
                            continue
                    print("Preparing data for: ", DOWNSTREAM_FOLDER)
                    dataloader = get_data(DATA, tokenizer, data)
                    run_and_write_uncertainty(modelwrapper, dataloader, DOWNSTREAM_FOLDER, device, DATA, MODEL)
                    print()
                    
    print("Completed.")
    