import nltk
from nltk.corpus import wordnet as wn
import random
from random import randint
from itertools import chain, compress
import numpy as np
import string
import json
import names


pos_tags = None

prefix = "" #"/tmp/svm-erda/"

# Load PPDB synonyms
with open('../../Resources/ppdb_synonyms.json') as json_file:
    clean_ppdb_synonyms = json.load(json_file)
    
with open('../../Resources/ppdb_synonyms_xxxl.json') as json_file:
    clean_ppdb_synonyms_XL = json.load(json_file)
    
with open('../../Resources/ppdb_synonyms_xxxl_nopostag.json') as json_file:
    clean_ppdb_synonyms_XXL = json.load(json_file)
    

NO_MATCH = set()
no_match_counter = 0
word_counter = 0
weird_cases = set()


ones = {
    0: '', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
    7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
    13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
    17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}
tens = {
    2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty',
    7: 'seventy', 8: 'eighty', 9: 'ninety'}
illions = {
    1: 'thousand', 2: 'million', 3: 'billion', 4: 'trillion', 5: 'quadrillion',
    6: 'quintillion', 7: 'sextillion', 8: 'septillion', 9: 'octillion',
    10: 'nonillion', 11: 'decillion'}

"""
Leet speak letter perturbation based on https://simple.wikipedia.org/wiki/Leet, excluding the space > 0.
"""
leet_letter_mappings = {
    "!": "1",
    "A": "4",
    "B": "8",
    "E": "3",
    "G": "6",
    "I": "1",
    "O": "0",
    "S": "5",
    "T": "7",
    "X": "8",
    "Z": "2",
    "a": "@",
    "b": "6",
    "e": "3",
    "g": "9",
    "h": "4",
    "i": "1",
    "l": "1",
    "o": "0",
    "s": "5",
    "t": "7",
    "z": "2"
}


def say_number(i):
    """
    Convert an integer in to it's word representation.

    say_number(i: integer) -> string
    """
    if i < 0:
        return _join('negative', _say_number_pos(-i))
    if i == 0:
        return 'zero'
    return _say_number_pos(i)


def _say_number_pos(i):
    if i < 20:
        return ones[i]
    if i < 100:
        return _join(tens[i // 10], ones[i % 10])
    if i < 1000:
        return _divide(i, 100, 'hundred')
    for illions_number, illions_name in illions.items():
        if i < 1000**(illions_number + 1):
            break
    return _divide(i, 1000**illions_number, illions_name)


def _divide(dividend, divisor, magnitude):
    return _join(
        _say_number_pos(dividend // divisor),
        magnitude,
        _say_number_pos(dividend % divisor),
    )


def _join(*args):
    return '-'.join(filter(bool, args))

def convert_to_case(old, new):
    global weird_cases
    if old.isupper():
        return new.upper()
    if old.islower():
        return new.lower()
    if old.istitle():
        return new.title()
    weird_cases.add(old)
    return new

def load_text(label):
    global text_label
    global pairwise
    text_label = label
    if type(text_label) == list:
        pairwise = True
    else:
        pairwise = False

def load_twitter_ids(dataset):
    global vocab
    global twitter_ids
    global text_label
    global pairwise
    vocab = set()
    twitter_ids = set()
    if not pairwise:
        for line in dataset[text_label]:
            for word in line.split():
                vocab.add(word.lower())
                if word[0] == '@':
                    twitter_ids.add(word)
    else:
        for l in text_label:
            for line in dataset[l]:
                for word in line.split():
                    vocab.add(word.lower())
                    if word[0] == '@':
                        twitter_ids.add(word)   
                


def find_replacement(word, pos=''):
    global twitter_ids
    
    # Get list of appropriate twitter aliases? and names?
    # Get list of punctuation
    quotes = [ "'", "''", "`", "``", '"']
    brackets = ["(", ")", "{", "}", "[", "]", '/']
    punct = [ '.', '!', '?', ',']
    breaks = ['-', '--', ',', ':', ';']
    
    if word[:12] == "http://t.co/" : ##URL: 
        return  word[:-8] + ''.join(random.choice(string.ascii_letters + string.digits) for i in range(8))
    
    if word[:13] == "https://t.co/" : ##URL:
        return  word[:-8] + ''.join(random.choice(string.ascii_letters + string.digits) for i in range(8))
    
    if word[0] == '#':
        return('#' + find_replacement(word[1:]))
    
    if word[0] == '@': # twitter Id
        return random.choice([t for t in twitter_ids if t != word])
    
    if pos == 'DT':
        dets = ['a', 'an', 'the', 'this', 'that']
        return convert_to_case(word, random.choice([d for d in dets if d != word.lower()]))
    
    if pos == 'WDT':
        wdts = ['that', 'what', 'whatever', 'which', 'whichever']
        return convert_to_case(word, random.choice([d for d in wdts if d != word.lower()]))
    
    # if pos == 'PRP$':
    #     prps = ['her', 'his', 'mine', 'my', 'our', 'ours', 'their', 'your']
    #     return random.choice([d for d in prps if d != word.lower()])
    
    # if pos == 'PRP':
    #     prps = ['hers', 'herself', 'him', 'himself', 'hisself', 'it', 'itself', 'me', 'myself', 'one', 'oneself', 'ours', 'ourselves', 'ownself', 'she', 'theirs', 'them', 'themselves', 'they', 'us']
        
    if pos == 'NNP': # Proper noun
        if word[-2:] == "'s":
            return convert_to_case(word[:-2], random.choice([names.get_first_name(), names.get_last_name()])) + "'s'"
        else:
            return convert_to_case(word, random.choice([names.get_first_name(), names.get_last_name()]))
    
    
    if word in quotes:
        return random.choice([d for d in quotes if d != word.lower()])
    
    if word in brackets:
        return random.choice([d for d in brackets if d != word.lower()])
    
    if word in punct:
        return random.choice([d for d in punct if d != word.lower()])
    
    if word in breaks:
        return random.choice([d for d in breaks if d != word.lower()])
    
    if word.isnumeric():
        return say_number(int(word))
        
    # Collect wordnet synonyms
    options_wn = [ w.replace("_", "-") for w in list(chain.from_iterable([syn.lemma_names() for syn in wn.synsets(word.lower(), pos=get_wordnet_pos(pos))])) if w != word]
    
    if options_wn == []:
        options_wn = [ w.replace("_", "-") for w in list(chain.from_iterable([syn.lemma_names() for syn in wn.synsets(word.lower())])) if w != word]
    
    # Collect synonyms from PPDB
    if pos != '':
        try:
            options_ppdb = clean_ppdb_synonyms[word.lower()][pos]
        except KeyError:
            options_ppdb = []
    else:
        try:
            options_ppdb = clean_ppdb_synonyms_XXL[word.lower()]
        except KeyError:
            options_ppdb = []
            
    # Babelnet?? -- REQ PYTHON 3.8
        
    full_set = options_wn + options_ppdb
    
    try:
        return convert_to_case(word,random.choice(full_set))
        
    except IndexError:
        
        if word[-1] in breaks:
            return find_replacement(word[:-1]) + word[-1]
            
        if word[0] in breaks:
            return word[0] + find_replacement(word[1:])
        
        if word[-1] in punct:
            return find_replacement(word[:-1]) + word[-1]
                    
        if word[0] in punct:
            return word[0] + find_replacement(word[1:])
        
        if word[-1] in quotes:
            return find_replacement(word[:-1]) + word[-1]
            
        if word[0] in quotes:
            return word[0] + find_replacement(word[1:])
        
        if word[-1] in brackets:
            return find_replacement(word[:-1]) + word[-1]
            
        if word[0] in brackets:
            return word[0] + find_replacement(word[1:])
        
        if word[-1] == "%":
            return find_replacement(word[:-1]) + '%'
        
        ### Try to parse by hyphens
        if '-' in word:
            parts = word.split('-')
            for i,p in enumerate(parts):
                n = find_replacement(p)
                if n != p:
                    return '-'.join(parts[:i] + [n] + parts[i+1:])

                
        if '/' in word:
            parts = word.split('/')
            for i,p in enumerate(parts):
                n = find_replacement(p)
                if n != p:
                    return '/'.join(parts[:i] + [n] + parts[i+1:])
                
        if '.' in word:
            parts = word.split('.')
            for i,p in enumerate(parts):
                if p != '':
                    n = find_replacement(p)
                    if n != p:
                        return '.'.join(parts[:i] + [n] + parts[i+1:])
    

        
        # #### Try to parse TextLikeThis
        # if len(word) > 1 and not word.isupper() and any(ele.isupper() for ele in word):
        #     parts = re.findall('[a-zA-Z][^A-Z]*', word)
        #     for i,p in enumerate(parts):
        #         if p != '':
        #             n = find_replacement(p)
        #             if n != p:
        #                 return ''.join(parts[:i] + [n] + parts[i+1:])
        
        ## check less good fits
        try:
            return convert_to_case(word,random.choice(clean_ppdb_synonyms_XL[word.lower()][pos]))
        except (KeyError, IndexError):
            try:
                return convert_to_case(word,random.choice(clean_ppdb_synonyms_XXL[word.lower()]))
            except ( KeyError, IndexError) :
                if word[-3:] == 'ish':
                    return find_replacement(word[:-3]) + 'ish'
                if word[-4:] == 'ness':
                    return find_replacement(word[:-4]) + 'ness'
                if word[-4:] == 'less':
                    return find_replacement(word[:-4]) + 'less'
                # if word.istitle(): # implies proper noun
                #     return random.choice([names.get_first_name(), names.get_last_name()])
                return word
            
    

def wordswap(iterable):
    global pos_tags
    global NO_MATCH
    global no_match_counter
    global word_counter
    out = []
    for i,x in enumerate(iterable):
        word_counter += 1
        new = find_replacement(x, pos_tags[i])
        if new == x:
            NO_MATCH.add((x, pos_tags[i]))
            no_match_counter += 1
        out.append(new)
                
    return out

def random_wordswap(iterable):
    return [ find_random(x) for x in iterable]

def find_random(word):
    global vocab
    return convert_to_case(word, random.choice([w for w in vocab if w != word.lower()]))

# pos_tags = None

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

def insert_char(string, index, chartoadd):
    return string[:index] + chartoadd + string[index:]

def swap_char(string, index, chartoadd):
    return string[:index] + chartoadd + string[index+1:]

def charinsert(iterable):
    return [insert_char(x, random.randint(0,len(x)), random.choice(string.ascii_letters)) for x in iterable]

def random_charswap(iterable):
    return [swap_char(x, random.randint(0,len(x)), random.choice(string.ascii_letters)) for x in iterable]

def realistic_charswap(iterable):
    return [butterfinger(x) for x in iterable]

def butterfinger(text,errors=1,keyboard='querty'):
    ### Adapted from https://github.com/alexyorke/butter-fingers/

    keyApprox = {}

    if keyboard == "querty": ## removed original word
        keyApprox['q'] = "wasedzx"
        keyApprox['w'] = "qesadrfcx"
        keyApprox['e'] = "wrsfdqazxcvgt"
        keyApprox['r'] = "etdgfwsxcvgt"
        keyApprox['t'] = "ryfhgedcvbnju"
        keyApprox['y'] = "tugjhrfvbnji"
        keyApprox['u'] = "yihkjtgbnmlo"
        keyApprox['i'] = "uojlkyhnmlp"
        keyApprox['o'] = "ipklujm"
        keyApprox['p'] = "lo['ik"

        keyApprox['a'] = "qszwxwdce"
        keyApprox['s'] = "wxadrfv"
        keyApprox['d'] = "ecsfaqgbv"
        keyApprox['f'] = "dgrvwsxyhn"
        keyApprox['g'] = "tbfhedcyjn"
        keyApprox['h'] = "yngjfrvkim"
        keyApprox['j'] = "hknugtblom"
        keyApprox['k'] = "jlinyhn"
        keyApprox['l'] = "okmpujn"

        keyApprox['z'] = "axsvde"
        keyApprox['x'] = "zcsdbvfrewq"
        keyApprox['c'] = "xvdfzswergb"
        keyApprox['v'] = "cfbgxdertyn"
        keyApprox['b'] = "vnghcftyun"
        keyApprox['n'] = "bmhjvgtuik"
        keyApprox['m'] = "nkjloik"
        keyApprox[' '] = " "
    else:
        print("Keyboard not supported.")
  
    if errors != 1:
        print("Can only make one error per text")
        return text
    
    error_idx = randint(0,len(text)) ### Choose a random letter in the text
    buttertext = ""
    for i,letter in enumerate(text):
        lcletter = letter.lower()
        if not lcletter in keyApprox.keys():
            newletter = lcletter
        else:
            if i == error_idx:
                newletter = random.choice(keyApprox[lcletter])
            else:
                newletter = lcletter
        # go back to original case
        if not lcletter == letter:
            newletter = newletter.upper()
        buttertext += newletter

    return buttertext


def convert_to_leet(word):
    global leet_letter_mappings
    out = ""
    for l in word:
        if l in leet_letter_mappings.keys():
            out += leet_letter_mappings[l]
        else:
            out += l
    return out

def insert_leet(iterable):
    return [convert_to_leet(x) for x in iterable]


def obscure_less(mask, to_remove, pipe_idx=None):
    old_masked = np.array(list(compress(range(len(mask)), mask)))
    if pipe_idx:
        old_masked = np.setdiff1d(old_masked, [pipe_idx]) ## Ensure pipe is not perturbed
 
    try:
        removed = np.random.choice(old_masked, size=to_remove, replace=False)
    except ValueError:
        return mask
            
        
    if len(removed) > 0:
        total_to_mask = np.setdiff1d(old_masked, removed)
        
        # Create new mask
        new_mask = np.zeros(len(mask), dtype=int)
        new_mask[total_to_mask] = 1
        new_mask.astype(bool)
    
        return new_mask
    else:
        print("OOOPS")
        return mask


def create_random_masks(example):
    noisy = {}
    pipe_idx = None
    has_pipe = False
    if "|" in example:
        has_pipe = True
        pipe_idx = example.index("|")

    ### Start with everything
    prop=1
    mask = np.ones(len(example), dtype=int)
    mask.astype(bool)
    updated_lasttime= True
    if has_pipe:
        input_len = len(mask) - 1
    else:
        input_len = len(mask)

    
    for additional in [.05,.05,.1,.1,.2,.25, 0.15, 0.05]:

        prop -= additional
        if not updated_lasttime:
            amt = old + additional
        else:
            amt = additional
            
        to_remove = round(amt * input_len)
        
        if to_remove == 0:
            updated_lasttime = False
            old = additional
        else:
            updated_lasttime = True
            mask = obscure_less(mask, to_remove, pipe_idx)
            
        noisy[f'random_token_{prop*100:02.0f}'] = ' '.join(np.where(mask, '{TOKEN}' , example))    
        noisy[f'random_charswap_{prop*100:02.0f}'] = ' '.join(np.where(mask, random_charswap(example) , example))    
        noisy[f'random_synonym_{prop*100:02.0f}'] = ' '.join(np.where(mask, wordswap(example) , example))  
        noisy[f'random_PATTERN_{prop*100:02.0f}'] = ' '.join([str(bool(t)) for t in mask])    
        noisy[f'random_butterfingers_{prop*100:02.0f}'] = ' '.join(np.where(mask, realistic_charswap(example) , example))
        noisy[f'random_wordswap_{prop*100:02.0f}'] = ' '.join(np.where(mask, random_wordswap(example) , example))
        noisy[f'random_charinsert_{prop*100:02.0f}'] = ' '.join(np.where(mask, charinsert(example) , example))
        noisy[f'random_l33t_{prop*100:02.0f}'] = ' '.join(np.where(mask, insert_leet(example) , example))   
        
    return noisy
        


def match_pos_token_to_original(pos_tokens, raw_orig, pos_tags):
    orig = []
    for word in raw_orig:
        if word != "":
            orig.append(word.strip())
    pos_idx = 0
    last_pos_idx = 1
    orig_idx = 0
    
    orig_to_pos_mapping = {}
    orig_idx2token = {}

    while pos_idx < len(pos_tokens) and orig_idx < len(orig):

        current_orig = orig[orig_idx]
        current_pos = pos_tokens[pos_idx]
        orig_to_pos_mapping[orig_idx] = [pos_idx]
        
        pos_idx += 1
        orig_idx2token[orig_idx] = current_orig
        if current_pos != current_orig:			
            combined = current_pos
            last_pos_idx = pos_idx
            while last_pos_idx < len(pos_tokens):
                next_part = pos_tokens[last_pos_idx]				
                combined += next_part
                orig_to_pos_mapping[orig_idx].append(last_pos_idx)
                if combined == current_orig:					
                    pos_idx = last_pos_idx + 1
                    break
                else:
                    last_pos_idx += 1

        orig_idx += 1
        
    pos_to_drop = ["$", '', "(", ")", ",", "#", "POS", "--", ".", ":", "''", '``']

    new_pos_tags = []
    for k in orig_to_pos_mapping.keys():
        if len(orig_to_pos_mapping[k]) == 1:
            new_pos_tags.append(pos_tags[orig_to_pos_mapping[k][0]])
        else:
            to_add = []
            for i in orig_to_pos_mapping[k]:
                if pos_tags[i] not in pos_to_drop:
                    to_add.append(pos_tags[i])
            if len(to_add) == 1:
                new_pos_tags.append(to_add[0])
            else:
                new_pos_tags.append('')
    
    return new_pos_tags

def create_human_masks(example, anno, is_heneutral=False, pairwise=False):
    global pos_tags
    noisy = {}
    
    if is_heneutral: ### In processing the neutral scores, the annotations do not correspong to the length of tokens anymore. Since they are all 0 anyway, we are just cutting them.
        anno = anno[:len(example)]
        
    ranked_anno = np.argsort(anno)
    
    if pairwise:
        ranked_anno = ranked_anno[:-1] # ignore last value as it is np.nan
    
    strategic_anno = [] # Hold strategic annotations
    for i, v in enumerate(anno):
        if v != 0.0:
            strategic_anno.append(v)
        elif pos_tags[i].startswith('J'): # ADJECTIVE
            strategic_anno.append(0.3) ### Minimum value in list is 0.333 so this will be after the list
        elif pos_tags[i].startswith('R'): # ADVERB
            strategic_anno.append(0.25)
        elif pos_tags[i].startswith('V'): # VERB
            strategic_anno.append(0.2)
        elif pos_tags[i].startswith('N'): # NOUN
            strategic_anno.append(0.1)
        else:
            strategic_anno.append(0.05)
            
    ranked_strat_anno = np.argsort(strategic_anno)
    if pairwise:
        ranked_strat_anno = ranked_strat_anno[:-1] # ignore last value as it is np.nan
    
    
    ##### Note when human annotation ends
    mask = (anno != 0)
    noisy[f'human-A_token'] = ' '.join(np.where(mask, '{TOKEN}', example))  
    noisy[f'human-A_charswap'] = ' '.join(np.where(mask, random_charswap(example), example)) 
    noisy[f'human-A_synonym'] = ' '.join(np.where(mask, wordswap(example), example)) 
    noisy[f'human-A_PATTERN'] = ' '.join([str(bool(t)) for t in mask])
    noisy[f'human-A_butterfingers'] = ' '.join(np.where(mask, realistic_charswap(example) , example))
    noisy[f'human-A_wordswap'] = ' '.join(np.where(mask, random_wordswap(example) , example))
    noisy[f'human-A_charinsert'] = ' '.join(np.where(mask, charinsert(example) , example))
    noisy[f'human-A_l33t'] = ' '.join(np.where(mask, insert_leet(example) , example))   
    

    for prop in [0.05,0.1,0.25,0.5,0.7,0.8,0.9,0.95]:
        
        ## Random fill
        mask = [True if ele in ranked_anno[-round(len(anno)*prop):] else False for ele in np.arange(len(anno))]
        noisy[f'human-R_token_{prop*100:02.0f}'] = ' '.join(np.where(mask, '{TOKEN}', example))  
        noisy[f'human-R_charswap_{prop*100:02.0f}'] = ' '.join(np.where(mask, random_charswap(example), example))   #' '.join(np.where(mask,  , example))     
        noisy[f'human-R_synonym_{prop*100:02.0f}'] = ' '.join(np.where(mask, wordswap(example), example))   #' '.join(np.where(mask, wordswap(example) , example))  
        noisy[f'human-R_PATTERN_{prop*100:02.0f}'] = ' '.join([str(bool(t)) for t in mask])   #' '.join(np.where(mask, wordswap(example) , example))  
        noisy[f'human-R_butterfingers_{prop*100:02.0f}'] = ' '.join(np.where(mask, realistic_charswap(example) , example))
        noisy[f'human-R_wordswap_{prop*100:02.0f}'] = ' '.join(np.where(mask, random_wordswap(example) , example))
        noisy[f'human-R_charinsert_{prop*100:02.0f}'] = ' '.join(np.where(mask, charinsert(example) , example))
        noisy[f'human-R_l33t_{prop*100:02.0f}'] = ' '.join(np.where(mask, insert_leet(example) , example))   
        
     
        ## Strategic Fill  
        
        mask = [True if ele in ranked_strat_anno[-round(len(anno)*prop):] else False for ele in np.arange(len(anno))]
        noisy[f'human-S_token_{prop*100:02.0f}'] = ' '.join(np.where(mask, '{TOKEN}', example))  
        noisy[f'human-S_charswap_{prop*100:02.0f}'] = ' '.join(np.where(mask, random_charswap(example), example))   #' '.join(np.where(mask,  , example))     
        noisy[f'human-S_synonym_{prop*100:02.0f}'] = ' '.join(np.where(mask, wordswap(example), example))   #' '.join(np.where(mask, wordswap(example) , example))  
        noisy[f'human-S_PATTERN_{prop*100:02.0f}'] = ' '.join([str(bool(t)) for t in mask])   #' '.join(np.where(mask, wordswap(example) , example))  
        noisy[f'human-S_butterfingers_{prop*100:02.0f}'] = ' '.join(np.where(mask, realistic_charswap(example) , example))
        noisy[f'human-S_wordswap_{prop*100:02.0f}'] = ' '.join(np.where(mask, random_wordswap(example) , example))
        noisy[f'human-S_charinsert_{prop*100:02.0f}'] = ' '.join(np.where(mask, charinsert(example) , example))
        noisy[f'human-S_l33t_{prop*100:02.0f}'] = ' '.join(np.where(mask, insert_leet(example) , example))   
        
    return noisy


def insert_random_noise(example):
    global pos_tags
    global tokens
    global text_label
    global pairwise
    
    if pairwise:
        full_text = ' | '.join([example[l] for l in text_label])
    else:
        full_text = example[text_label]
        
        
    nltk_tokens = nltk.word_tokenize(full_text)
    pos_tags = [val[1] for val in nltk.pos_tag(nltk_tokens)]
    
    tokens = full_text.split()
    
    pos_tags = match_pos_token_to_original(nltk_tokens, tokens, pos_tags)
        
    noise = create_random_masks(tokens)
    return example | noise

def insert_human_noise(example, is_heneutral=False):
    global pos_tags
    global text_label
    global pairwise
    
    if pairwise:
        full_text = ' | '.join([example[l] for l in text_label])
    else:
        full_text = example[text_label]
        
        
    nltk_tokens = nltk.word_tokenize(full_text)
    pos_tags = [val[1] for val in nltk.pos_tag(nltk_tokens)]
    
    tokens = full_text.split()
    pos_tags = match_pos_token_to_original(nltk_tokens, tokens, pos_tags)
    pos_to_save = {'pos_tags' : pos_tags}
    
    if pairwise:
        anno = np.abs(np.array(example['annotation_1'] + [np.nan] + example['annotation_2'])) ### use np.nan for seperator
    
    else:
        if type(example['annotations']) == list:
            anno = np.abs(np.array(example['annotations']).astype(float))
        else:
            anno = np.abs(np.array(example['annotations'].split()).astype(float)) ## Chose + or - to indicate if positive or negative word in SemEval
    
    noise = create_human_masks(tokens, anno, is_heneutral, pairwise)
    return example | noise | pos_to_save

def load_gradients(MODEL, DATA):
    global annos
    global model  
    model = MODEL # set global variable here for later functions
    
    annos_file = f"{prefix}Hotflip/{MODEL}-{DATA}.json"

    with open(annos_file) as json_file:
        annos = json.load(json_file)
    

def create_gradient_masks(example, anno, MODEL, pipe_idx=None):
    noisy = {}

    for prop in [0.05,0.1,0.25,0.5,0.7,0.8,0.9,0.95]:
        mask = [True if ele in anno[-round(len(anno)*prop):] else False for ele in np.arange(len(anno))]
        if pipe_idx:
            mask.insert( pipe_idx, False)
        #mask = [True if ele in anno[:round(len(anno)*prop)] else False for ele in np.arange(len(anno))]
        noisy[f'gradient-{MODEL}_token_{prop*100:02.0f}'] = ' '.join(np.where(mask, '{TOKEN}', example))  
        noisy[f'gradient-{MODEL}_charswap_{prop*100:02.0f}'] = ' '.join(np.where(mask, random_charswap(example), example))   #' '.join(np.where(mask,  , example))     
        noisy[f'gradient-{MODEL}_synonym_{prop*100:02.0f}'] = ' '.join(np.where(mask, wordswap(example), example))   #' '.join(np.where(mask, wordswap(example) , example))
        noisy[f'gradient-{MODEL}_PATTERN_{prop*100:02.0f}'] = ' '.join([str(bool(t)) for t in mask])  #' '.join(np.where(mask, wordswap(example) , example))     
        noisy[f'gradient-{MODEL}_butterfingers_{prop*100:02.0f}'] = ' '.join(np.where(mask, realistic_charswap(example) , example))
        noisy[f'gradient-{MODEL}_wordswap_{prop*100:02.0f}'] = ' '.join(np.where(mask, random_wordswap(example) , example))
        noisy[f'gradient-{MODEL}_charinsert_{prop*100:02.0f}'] = ' '.join(np.where(mask, charinsert(example) , example))
        noisy[f'gradient-{MODEL}_l33t_{prop*100:02.0f}'] = ' '.join(np.where(mask, insert_leet(example) , example))   
            
    return noisy

def insert_gradient_noise(example):
    global pos_tags
    global text_label
    global pairwise
    
    if pairwise:
        full_text = ' | '.join([example[l] for l in text_label])
    else:
        full_text = example[text_label]
    
    pipe_idx = None
        
        
    nltk_tokens = nltk.word_tokenize(full_text)
    pos_tags = [val[1] for val in nltk.pos_tag(nltk_tokens)]
    
    tokens = full_text.split()
    ind = example['index']
    anno = annos[str(ind)]
    if "|" in example:
        has_pipe = True
        pipe_idx = tokens.index("|")
        anno.remove(pipe_idx) ### Do not perturb the |!
    
    noise = create_gradient_masks(tokens, anno, model, pipe_idx)
    return example | noise

def __main__():
    exit()