'''
 Implementation module to reload models and evaluate them.
'''

import numpy as np
from typing import List, Tuple, Dict

from model import Model
import random

# import my code base
from stud.Utils import *
from stud.Dataset import *
from stud.Classifiers import *


# building of the models --> demand the construction to StudentModel Class
def build_model_b(device: str) -> Model:
    return StudentModel(device = device, task="b")

def build_model_ab(device: str) -> Model:
    return StudentModel(device = device, task="ab")  

def build_model_cd(device: str) -> Model:
    return StudentModel(device = device, task="cd")

class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
        ("service", 248),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else: 
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]) 
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0]) 
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds



class StudentModel(Model):
    """
    StudentModel class to run test.sh and evaluate the delivered models
    ...

    Attributes
    ----------
    train_param : dict
        a dict were all the training paramter as learning rate or loss weight are provided (here random values just to call the constructor of each model).
    device : str
        device to assign to each model and tensors
    task : str 
        actual task to perform --> use to control the behaviour of the prediction
    Model_A, Model_B, Model_C, Model_D : torch.nn.module
        all the different models
    """
    
    def __init__(self, device="cpu", task = 'b'):
    
        self.task   = task
        self.device = device
        self.train_param = {'freeze':False, 'lr':0.00005,'eps':1e-4, 'dropout':0.2,'loss_weight':None}  
        
        # load the needed models based on required task, and set to eval mode and to device.
        if task == "b":
            self.model_B = Classifier_B.load_from_checkpoint("./model/B-FOUR.ckpt",strict=False, train_param=self.train_param, mode="four")
            self.model_B.to(self.device)
            self.model_B.eval()
            
        if task == "ab":
            self.model_A = Classifier_A.load_from_checkpoint("./model/A-IOB.ckpt",strict=False,train_param=self.train_param, mode="iob")
            self.model_A.eval()
            self.model_A.to(self.device)            
            self.model_B = Classifier_B.load_from_checkpoint("./model/B-FOUR.ckpt",strict=False,train_param=self.train_param, mode="four")
            self.model_B.eval()
            self.model_B.to(self.device)
            
        if task == "cd":
            self.model_C = Classifier_C.load_from_checkpoint("./model/C.ckpt",strict=False,train_param=self.train_param)
            self.model_C.eval()
            self.model_C.to(self.device)            
            self.model_D = Classifier_D.load_from_checkpoint("./model/D-FOUR.ckpt", strict=False, train_param=self.train_param, mode="four")
            self.model_D.eval()
            self.model_D.to(self.device)      


    # general predict method
    def predict(self, samples: List[Dict]) -> List[Dict]:
    
        if self.task=="b":      return self.predict_b(samples)
        if self.task=="ab":     return self.predict_ab(samples)
        if self.task=="cd":     return self.predict_cd(samples)
        
        
    # predict method for b task
    def predict_b(self, samples: List[Dict]) -> List[Dict]:
        predictions = []
        
        for sample in samples:
            pred = {"targets":[]}
            
            for target in sample["targets"]:
                tokenized = encode_text(sample["text"],target[1])
                x = tokenized['input_ids']
                mask =  tokenized['attention_mask']
                res = self.model_B(x,mask,None)
                pred['targets'] += [ ( target[1] , idx2sent[res['pred'].item()] ) ]
                
            predictions  += [ pred ]
           
        return predictions


    # predict method for a+b task
    def predict_ab(self, samples: List[Dict]) -> List[Dict]:
        predictions = []
        
        for sample in samples:
            pred = {"targets":[]}
            tokenized = encode_text(sample["text"])
            x = tokenized['input_ids']
            mask =  tokenized['attention_mask']
            res = self.model_A(x,mask,None)
            targets = decode_aspect(res['pred'].squeeze(), x.squeeze(), mask=mask.squeeze(), mode="iob")
            
            for target in targets:
                tokenized = encode_text(sample["text"],target)
                x = tokenized['input_ids']
                mask =  tokenized['attention_mask']
                res = self.model_B(x,mask,None)
                pred['targets'] += [ ( target , idx2sent[res['pred'].item()] ) ]
                
            predictions  += [ pred ]

        return predictions
    
    
    # predict method for c+d task
    def predict_cd(self, samples: List[Dict]) -> List[Dict]:
        predictions = [] 
        
        for sample in samples:
            pred = {"categories":[]}
            
            tokenized = encode_text(sample["text"])
            x = tokenized['input_ids']
            mask =  tokenized['attention_mask']
            res = self.model_C(x,mask,None)
            
            categories = []
            for i, x in enumerate(res['pred'].squeeze()):
                if x.item() == 1:
                    categories += [ idx2cat[i] ]
            
            for cat in categories:
                tokenized = encode_text(sample["text"],cat)
                x = tokenized['input_ids']
                mask =  tokenized['attention_mask']
                res = self.model_D(x,mask,None)

                pred['categories'] += [ ( cat , idx2sent[res['pred'].item()] ) ] 
      
            predictions  += [ pred ]
        
        return predictions
