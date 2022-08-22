'''
 This module contains all code for the dataset part of homework, collate_fn, encoding part, dataset, dataloader, ..
'''

# Libraries
from torch.utils.data import Dataset, DataLoader
from stud.Utils import *
from typing import *
import pytorch_lightning as pl
import json
import random


# ------------------ *** ------------------- #
# Dataset class for training and validation stage

class TextDataset(Dataset):
    """
    Dataset class to support training and validation
    ...

    Attributes
    ----------
    data : list[dict]
        a lis of dict, were each dict is a encoded sample (depending on the task)
    """
    
    # constructor
    def __init__(self, data: List[Dict], task:str = "a"):
        
        # different task needs different encoding
        if task=="a": self.data = encode_dataset_A(data)
        if task=="b": self.data = encode_dataset_B(data)
        if task=="c": self.data = encode_dataset_C(data)
        if task=="d": self.data = encode_dataset_D(data)


    # Load the dataset given the filepath.
    @classmethod
    def from_disk(cls, data_path: str, task:str = "a"):

        data = []
        with open(data_path) as json_file:
            data = json.load(json_file)

        return cls(data, task=task)


    # Merge the dataset with another one
    def merge(self, textdataset: Dataset):
      self.data += textdataset.data
      random.shuffle(self.data)
      return self

    # Returns the length of the dataset.
    def __len__(self) -> int:
        return len(self.data)

    # Returns a sample from the dataset.
    def __getitem__(self, index: int) -> dict:
        return self.data[index]



# ------------------ *** ------------------- #
# Data Module leveraging pytorch lightning


class TextDataModule(pl.LightningDataModule):
    """
    DataModule lightning class to support training and validation
    ...

    Attributes
    ----------
    train_dataset : torch.utils.data.Dataset
        dataset for training
    validation_dataset : torch.utils.data.Dataset
        dataset for validation
    test_dataset : torch.utils.data.Dataset
        dataset for test
    collate_fn :  Callable
        specific function for collate batches (each task needs different collate strategy)
    batch_size : int
        size of the batch for train and validation
    """
    
    # constructor
    def __init__(self, train_dataset: Dataset = None, val_dataset: Dataset = None, test_dataset: Dataset = None, batch_size: int = 16, collate_fn: callable = None) -> None:
        super().__init__()

        self.train_dataset = train_dataset
        self.validation_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.collate_fn = collate_fn


    # Methods that return dataloaders
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=True, drop_last=True,collate_fn=self.collate_fn )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.validation_dataset, batch_size=self.batch_size,shuffle=False, drop_last=True,collate_fn=self.collate_fn )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=False, drop_last=True,collate_fn=self.collate_fn )




# ------------------ *** ------------------- #
# Collate methods for batches (for each task)

# Collate methods for task  A
def collate_A(batch: Dict):

    # reverse list of dict to dict of list
    res = {key: [] for key in batch[0]}
    {res[key].append(sub[key]) for sub in batch for key in sub} 

    # stack lists to unified tensor
    res['input_ids'] = torch.stack(res['input_ids'])
    res['token_type_ids'] = torch.stack(res['token_type_ids'])
    res['attention_mask'] = torch.stack(res['attention_mask'])

    res['aspect_label_io'] = torch.stack(res['aspect_label_io'])
    res['aspect_label_iob'] = torch.stack(res['aspect_label_iob'])
    res['aspect_label_e2e'] = torch.stack(res['aspect_label_e2e'])

    return res
  

# Collate methods for task  B
def collate_B(batch: Dict):

    # reverse list of dict to dict of list
    res = {key: [] for key in batch[0]}
    {res[key].append(sub[key]) for sub in batch for key in sub} 

    # stack lists to unified tensor
    res['input_ids'] = torch.stack(res['input_ids'])
    res['token_type_ids'] = torch.stack(res['token_type_ids'])
    res['attention_mask'] = torch.stack(res['attention_mask'])

    res['sentiment_bin'] = torch.stack(res['sentiment_bin'])
    res['sentiment_four'] = torch.stack(res['sentiment_four'])

    return res


# Collate methods for task  C
def collate_C(batch: Dict):

    # reverse list of dict to dict of list
    res = {key: [] for key in batch[0]}
    {res[key].append(sub[key]) for sub in batch for key in sub} 

    # stack lists to unified tensor
    res['input_ids'] = torch.stack(res['input_ids'])
    res['token_type_ids'] = torch.stack(res['token_type_ids'])
    res['attention_mask'] = torch.stack(res['attention_mask'])

    res['categories_label'] = torch.stack(res['categories_label'])

    return res


# Collate methods for task  D
def collate_D(batch: Dict):

    # reverse list of dict to dict of list
    res = {key: [] for key in batch[0]}
    {res[key].append(sub[key]) for sub in batch for key in sub} 

    # stack lists to unified tensor
    res['input_ids'] = torch.stack(res['input_ids'])
    res['token_type_ids'] = torch.stack(res['token_type_ids'])
    res['attention_mask'] = torch.stack(res['attention_mask'])

    res['sentiment_bin'] = torch.stack(res['sentiment_bin'])
    res['sentiment_four'] = torch.stack(res['sentiment_four'])

    return res




# ------------------ *** ------------------- #
# Encoding methods for datasets (for each task)


# given a raw dataset, read from file, return the encoded dict for task A
def encode_dataset_A (data: Dict):

    # loop tougth the samples
    for sample in data:
        # check for category field in laptop dataset
        sample['categories'] = sample['categories'] if 'categories' in sample.keys() else []
        
        # text encoding -> x
        tokenized = encode_text(sample['text'])
        sample['input_ids'] = tokenized['input_ids'].squeeze()
        sample['token_type_ids'] = tokenized['token_type_ids'].squeeze()
        sample['attention_mask'] = tokenized['attention_mask'].squeeze()

        # encode the aspect labels -> y
        sample['aspect_label_io']  = encode_aspect(  sample['targets'] , sample["input_ids"], sample['attention_mask'],mode ="io")
        sample['aspect_label_iob'] = encode_aspect(  sample['targets'] , sample["input_ids"], sample['attention_mask'],mode ="iob") 
        sample['aspect_label_e2e'] = encode_aspect(  sample['targets'] , sample["input_ids"], sample['attention_mask'],mode ="e2e") 
      
    return data


# given a raw dataset, read from file, return the encoded dict for task B
def encode_dataset_B (data: Dict):

    # new returning dataset
    new_dataset = []

    # loop tougth the samples
    for sample in data:
        # check for category field in laptop dataset
        sample['categories'] = sample['categories'] if 'categories' in sample.keys() else []

        # create a new sample for each target (and sentiment)
        for element in  sample['targets'] :
        
            # encoding the text/aspect -> x (sentence pair classification)
            tokenized = encode_text(sample['text'],element[1])
            sample['input_ids'] = tokenized['input_ids'].squeeze()
            sample['token_type_ids'] = tokenized['token_type_ids'].squeeze()
            sample['attention_mask'] = tokenized['attention_mask'].squeeze()

            new_sample = copy.deepcopy(sample)

            # encode the sentiment labels -> y
            new_sample ['sentiment_four'] = torch.tensor(sent2idx[element[2]])
            new_sample ['sentiment_bin'] = sent2pt(element[2])            

            new_dataset += [ new_sample ]

    return new_dataset


# given a raw dataset, read from file, return the encoded dict for task C
def encode_dataset_C (data: Dict):

    # loop tougth the samples
    for sample in data:
        # check for category field in laptop dataset
        sample['categories'] = sample['categories'] if 'categories' in sample.keys() else []
        
        # check for category field in laptop dataset
        tokenized = encode_text(sample['text'])
        sample['input_ids'] = tokenized['input_ids'].squeeze()
        sample['token_type_ids'] = tokenized['token_type_ids'].squeeze()
        sample['attention_mask'] = tokenized['attention_mask'].squeeze()

        # encode the category onehot vector -> y
        sample['categories'] = sample['categories'] if 'categories' in sample.keys() else []
        sample['categories_label'] = encode_category (sample['categories'])

    return data


# given a raw dataset, read from file, return the encoded dict for task D
def encode_dataset_D (data: Dict):

    # new returning dataset
    new_dataset = []

    # loop tougth the samples
    for sample in data:
        sample['categories'] = sample['categories'] if 'categories' in sample.keys() else []
      
        # create a new sample for each category (and sentiment)
        for element in  sample['categories'] :
        
            # encoding the text/category -> x (sentence pair classification)
            tokenized = encode_text(sample['text'],element[0])
            sample['input_ids'] = tokenized['input_ids'].squeeze()
            sample['token_type_ids'] = tokenized['token_type_ids'].squeeze()
            sample['attention_mask'] = tokenized['attention_mask'].squeeze()

            new_sample = copy.deepcopy(sample)

            # encode the sentiment labels -> y
            new_sample ['sentiment_four'] = torch.tensor(sent2idx[element[1]])
            new_sample ['sentiment_bin'] = sent2pt(element[1])            

            new_dataset += [ new_sample ]

    return new_dataset