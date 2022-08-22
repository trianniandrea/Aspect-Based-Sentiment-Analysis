'''
 This module contains all the utility methods and the 'global' variables for the homework, both for training stage and inference purposes
'''

# Libraries
from transformers import AutoTokenizer
import torch
import copy
from typing import *

# ------------------ *** ------------------- #
# Definition of standard dictionaries to translate label to index and index to label, for each task


# End to End ABSA -- token encoding -- begin encapsulates the sentiment
e2e     = {"pad":0, "outside":1, "inside":2, "positive":3, "negative":4, "neutral":5, "conflict":6}
e2e_rev = {0:"pad", 1:"outside", 2:"inside", 3:"positive", 4:"negative", 5:"neutral", 6:"conflict"}

# BIO / IO for task-A
tok2idx = {"pad":0, "outside":1, "inside":2, "begin":3}

# Sentiment Task-B,D
sent2idx = {"positive":0, "negative":1, "neutral":2, "conflict":3}
idx2sent = {0:"positive",1:"negative",2: "neutral",3:"conflict"}

# Another encoding strategy forentiment Task-B,D
def sent2pt(sentiment):
    if sentiment == "positive":  return torch.tensor([0.0, 1.0])
    if sentiment == "negative":  return torch.tensor([1.0, 0.0])
    if sentiment == "conflict":  return torch.tensor([1.0, 1.0])
    if sentiment == "neutral":   return torch.tensor([0.0, 0.0])

def pt2sent(tensor):
    if tensor[0] > 0.5  and tensor[1] > 0.5:     return "conflict"
    elif tensor[0] <= 0.5  and tensor[1] <= 0.5: return "neutral"
    elif tensor[0] > tensor[1]:                  return "negative" 
    elif tensor[0] <= tensor[1]:                 return "positive"

# Category Task-C
cat2idx = {"anecdotes/miscellaneous":0, "food":1, "service":2, "price":3, "ambience":4}
idx2cat = {0:"anecdotes/miscellaneous", 1:"food", 2:"service", 3:"price", 4:"ambience"}

# Loading from disk of tokenizer and dictionary
try:
    tokenizer= AutoTokenizer.from_pretrained('./model/pre_trained')
except:
    tokenizer= AutoTokenizer.from_pretrained('../model/pre_trained')


# Max length of input tensor for all the models.
MAX_LEN=80




# ------------------ *** ------------------- #
# Definition of utility method for encoding and so on..


# Given a list of target aspects, a vectorized text, and its mask, return a vector with the desidered encoding stategy (set by 'mode' parameter).
# The returning vector feeds the models for task A and E2E absa
def encode_aspect (targets: List[object], text: torch.Tensor, mask: torch.Tensor, mode: str = "iob"):

    # At the start fill the vector w/ outside token, then applying element-wise multiplication w/ mask i implicitly insert the pad tokens
    labels = [ tok2idx["outside"] for _ in range(len(text))]
    labels = [ a*b for a,b in zip(mask,labels)]

    # now insert the correct token labels for each target
    for target in targets:
    
        # target vector to string 
        t = tokenizer(target[1],add_special_tokens=False)['input_ids']
        cursor = 0
        
        # search each target inside the text
        for i in range(len(text)):
        
            # first token of the target found --> check now until the end token
            if text[i] == t[cursor]:
                cursor += 1
                
                # end token found --> i can insert correct labels now, leveraging the index
                if cursor == len(t):
                    labels[i-cursor+1:i+1] = [ tok2idx["inside"] for _ in range(len(t))]
                    if mode == "iob": labels[i-cursor+1] = tok2idx["begin"]
                    if mode == "e2e": labels[i-cursor+1] = e2e[target[2]]
                    cursor = 0
                    
            # no token matching
            else:
                cursor = 0

    # return tensor
    return torch.LongTensor( labels )


# given an output vector, and the vectorized text, return a list of founded aspect, as strings.
def decode_aspect (y_model: torch.Tensor, text: torch.Tensor, mask: torch.Tensor, mode: str = "iob"):

    # logically remove pad token from model output
    y_model = [ a*b for a,b in zip(mask.tolist(),y_model.tolist()) ]

    # list to return
    out_target = [ ]
    i = 0
    
    # loop trought the output trying to find interesting token
    while i < len(y_model):

        # token found, each mode has its trigger rule
        if (mode=="iob" and y_model[i] == tok2idx["begin"]) or (mode=="io" and y_model[i] == tok2idx["inside"]) or (mode=="e2e" and y_model[i]>=3):

            # catch sentiment (if needed) and first token of aspect
            target = [text[i]]
            sentiment = e2e_rev[y_model[i]]

            # catch remaining token of aspect
            for a in range(i+1, len(y_model)):
              if y_model[a] != tok2idx["inside"]: i = a; break;
              target += [text[a]]
              
            # convert token ids to string
            str_target = tokenizer.decode(torch.as_tensor(target).type(torch.IntTensor),add_special_tokens=False)
            
            # add to output
            if mode=="e2e": out_target += [(str_target,sentiment)]
            else:           out_target += [str_target]
                
        # token not found -> go forward
        else:
            i += 1
            target = []    

    return out_target


# Function to encode the category to a one-hot vector
def encode_category (categories: List):

    # start w/ all 0
    labels = [ 0 for _ in range(len(cat2idx.keys())) ]
    
    # fill present categories w/ 1
    for cat, sent in categories:
        labels [ cat2idx[cat] ] = 1
        
    # return a tensor
    return torch.FloatTensor(labels)



# Encode the raw string text and return a fixed length tensor of vocabulary indexes
def encode_text(text1: str, text2: str = None):
    # standard encoding
    if text2==None:     return tokenizer(text1, padding='max_length', truncation=True, return_tensors="pt", max_length=MAX_LEN)
    # sentence pair classification encoding (task b,d -- sentence/aspect)
    else:               return tokenizer(text1, text2, padding='max_length', truncation=True, return_tensors="pt", max_length=MAX_LEN)




# ------------------ *** ------------------- #
# evaluate the extraction performance [f1 is micro in this case], used in task A and e2e
# taken from evaluate.py, and adapted.

def evaluate_extraction(samples, predictions_b):
    scores = {"tp": 0, "fp": 0, "fn": 0}
    for label, pred in zip (samples, predictions_b):
        pred_terms = {term_pred for term_pred in pred}
        gt_terms = {term_gt for term_gt in label}

        scores["tp"] += len(pred_terms & gt_terms)
        scores["fp"] += len(pred_terms - gt_terms)
        scores["fn"] += len(gt_terms - pred_terms)

    precision = 100 * scores["tp"] / (scores["tp"] + scores["fp"] + 0.0000001)
    recall = 100 * scores["tp"] / (scores["tp"] + scores["fn"] + 0.0000001)
    f1 = 2 * precision * recall / (precision + recall + 0.0000001)

    return {'precision':precision, 'recall':recall, 'f1':f1/100.0}
    
