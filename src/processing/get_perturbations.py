from ..utils.perturbation_utils import *
from datasets import Dataset, load_from_disk, DatasetDict
import time
import sys

### Load data

if __name__ == "__main__": 
    DATA = sys.argv[1]

    if DATA == 'hatexplain_neutral':
        is_heneutral = True
    else:
        is_heneutral = False
        

    from_path_dataset = f"./Data/Clean/{DATA}"

    dataset = load_from_disk(from_path_dataset)['test'] #[:50] 
    # dataset = Dataset.from_dict(dataset)

    if "_" in DATA:
        from_path_dataset = f"./Data/Clean/{DATA.split('_')[0]}"

        full_dataset = load_from_disk(from_path_dataset)['test']
    else:
        full_dataset = dataset
        
    if DATA == "esnli":
        text_label = [ "text_1", "text_2"]
    else:
        text_label = "text"
        
    load_text(text_label)
    load_twitter_ids(full_dataset) ### Ensure we are looking at the entire dataset, not just that subset.
    print(DATA)
    print("Inserting Random Noise")
    new_dataset = dataset.map(insert_random_noise)
    print("Inserting Human Noise")
    new_dataset = new_dataset.map(insert_human_noise, is_heneutral)
    for MODEL in ['bert', 'electra', 'roberta', 'gpt2-medium']:
        print(f"Inserting {MODEL} Noise")
        load_gradients(MODEL, DATA)
        new_dataset = new_dataset.map(insert_gradient_noise)
        
    path_dataset = f"../../Data/Noise/{DATA}"
    new_dataset.cache_files
    new_dataset.save_to_disk(path_dataset)

