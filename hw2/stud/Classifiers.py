'''
 This module contains all the N.N. classifiers of the homework.
 There is a model class for each task, and a model class for e2e absa.
'''

# Libraries
import torchmetrics as tm
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup, BertForTokenClassification,BertForSequenceClassification
import torch.nn as nn
import torch
import pytorch_lightning as pl

from typing import *
from stud.Utils import *


class Classifier_E2E (pl.LightningModule):
    """
    End to End model for aspect based sentiment analysis based on bert + lstm
    Unified solution for task A+B
    ...

    Attributes
    ----------
    train_param : dict
        a dict were all the training paramter as learning rate or loss weight are provided.
    bert : torch.nn.module
        pretrained bert base cased from hugginface
    dropout : torch.nn.dropout 
        dropout between bert and lstm
    lstm1 : torch.nn.module
        a lstm bidirection cell
    fc1 : torch.nn.module
        the final linear layer of the network
    criterion : torch.nn.CrossEntropyLoss
        loss of the network
    """
    
    # constructor
    def __init__(self, train_param:Dict , path:str = "./model/pre_trained"):
        super().__init__()
        
        self.train_param = train_param
        self.bert = BertModel.from_pretrained(path)
        self.dropout = nn.Dropout(p=self.train_param['dropout'])
        self.lstm1 = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=self.bert.config.hidden_size, num_layers=1, dropout=self.train_param['dropout'], bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.bert.config.hidden_size*2, out_features=7)
        
        self.criterion = nn.CrossEntropyLoss(weight=self.train_param['loss_weight'], ignore_index=tok2idx["pad"])

        # freeze bert weight --> use it as a feature extrator
        if self.train_param['freeze']:
          for param in self.bert.parameters():
            param.requires_grad = False


    # single forward step 
    def forward(self, x: torch.Tensor, mask: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    
        output = self.bert(input_ids=x, attention_mask=mask)
        output, (h_t1, h_c) = self.lstm1(self.dropout(output.last_hidden_state))
        output = self.fc1(self.dropout(output))

        logits = torch.softmax(output, dim=2)
        pred  = torch.argmax(logits , dim=2)
        
        loss = 0
        if y is not None:
            loss= self.criterion(output.view(-1, logits.shape[-1]), y.view(-1))

        result = {'logits': logits, 'loss': loss, 'pred':pred}

        return result

    
    # lightning training step
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch['aspect_label_e2e'] )
        return forward_output['loss']


    # lightning validation step
    def validation_step(self,  batch: Tuple[torch.Tensor],  batch_idx: int):
        forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch['aspect_label_e2e'] )

        out_aspect = []; true_aspect = []
        for i, x in enumerate(forward_output['pred']):
            out_aspect  += [ decode_aspect(x , batch['input_ids'][i], batch['attention_mask'][i], mode="e2e") ]
            true_aspect += [ [(x[1],x[2]) for x in batch['targets'][i]] ]
            
        metric = evaluate_extraction( true_aspect, out_aspect)
        self.log('val_f1', metric["f1"], prog_bar=True)
        self.log('val_loss', forward_output['loss'], prog_bar=True)


    def test_step(self,batch: Tuple[torch.Tensor],batch_idx: int):
        pass


    # return the optimizer
    def configure_optimizers(self):
        return AdamW(self.parameters(), self.train_param['lr'], eps=self.train_param['eps'])
        
 
 
class Classifier_A (pl.LightningModule):
    """
    Model for aspect based sentiment analysis (TASK A) based on bert + linear layer
    ...

    Attributes
    ----------
    train_param : dict
        a dict were all the training paramter as learning rate or loss weight are provided.
    mode : str
        specify the ouput encoding io or iob
    bert : torch.nn.module
        pretrained bert base cased from hugginface
    dropout : torch.nn.dropout 
        dropout between bert and lstm
    classifier : torch.nn.module
        the final linear layer of the network
    criterion : torch.nn.CrossEntropyLoss
        loss of the network
    """
    
    # constructor
    def __init__(self, train_param:Dict, mode ="iob", path="./model/pre_trained"):
        super().__init__()

        self.train_param = train_param
        self.mode = mode
        self.bert = BertModel.from_pretrained(path)
        self.dropout = nn.Dropout(p=self.train_param['dropout'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, 4 if self.mode=="iob" else 3)
        self.criterion = nn.CrossEntropyLoss(weight=self.train_param['loss_weight'], ignore_index=tok2idx["pad"])

        # freeze bert weight --> use it as a feature extrator
        if self.train_param['freeze']:
          for param in self.bert.parameters():
            param.requires_grad = False


    # single forward step 
    def forward(self, x: torch.Tensor, mask: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        output = self.bert(input_ids=x, attention_mask=mask)
        output = self.classifier(self.dropout(output.last_hidden_state))

        logits = torch.softmax(output, dim=2)
        pred  = torch.argmax(logits , dim=2)
        
        loss = 0
        if y is not None:
            loss= self.criterion(output.view(-1, logits.shape[-1]), y.view(-1))

        result = {'logits': logits, 'loss': loss, 'pred':pred}

        return result


    # lightning training step
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.mode == "iob":
            forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch['aspect_label_iob'] )
        if self.mode == "io":
            forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch['aspect_label_io'] )

        return forward_output['loss']


    # lightning validation step
    def validation_step(self,  batch: Tuple[torch.Tensor],  batch_idx: int):
        if self.mode == "iob":
            forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch['aspect_label_iob'] )
        if self.mode == "io":
            forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch['aspect_label_io'] )

        out_aspect = []; true_aspect = []
        for i, x in enumerate(forward_output['pred']):
            out_aspect += [ decode_aspect(x , batch['input_ids'][i], batch['attention_mask'][i], mode=self.mode) ]
            true_aspect += [ [x[1] for x in batch['targets'][i]] ]
            
        metric = evaluate_extraction( true_aspect, out_aspect)
        self.log('val_f1', metric["f1"], prog_bar=True)
        self.log('val_loss', forward_output['loss'], prog_bar=True)


    def test_step(self,batch: Tuple[torch.Tensor],batch_idx: int):
        pass


    # return the optimizer
    def configure_optimizers(self):
        return AdamW(self.parameters(), self.train_param['lr'], eps=self.train_param['eps'])



class Classifier_B(pl.LightningModule):
    """
    A model for aspect based sentiment analysis (TASK B) based on bert + linear
    ...

    Attributes
    ----------
    train_param : dict
        a dict were all the training paramter as learning rate or loss weight are provided.
    mode : str
        'four' or 'bin' to select final output layer (see report)
    bert : torch.nn.module
        pretrained bert base cased from hugginface
    dropout : torch.nn.dropout 
        dropout between bert and lstm
    classifier : torch.nn.module
        the final linear layer of the network
    criterion : torch.nn.CrossEntropyLoss
        loss of the network
    val_f1 : torchmetrics
        provide macro f1 score during training
    """
    
    # constructor
    def __init__(self, train_param:Dict, mode = "four",path="./model/pre_trained" ):
        super().__init__()

        self.train_param = train_param
        self.mode=mode
        self.bert = BertModel.from_pretrained(path)
        self.dropout = nn.Dropout(p=self.train_param['dropout'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, 4 if self.mode=="four" else 2)
        self.criterion = nn.CrossEntropyLoss(self.train_param['loss_weight']) if self.mode=="four" else nn.BCELoss()
        self.val_f1 =  tm.F1(num_classes=4, average='macro')
 
        # freeze bert weight --> use it as a feature extrator 
        if self.train_param['freeze']:
          for param in self.bert.parameters():
            param.requires_grad = False


    # single forward step 
    def forward(self, x: torch.Tensor, mask: torch.Tensor, token_type: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        output = self.bert(input_ids=x, attention_mask=mask, token_type_ids = token_type)
        output = self.classifier(self.dropout(output.pooler_output))

        if self.mode == "four":
          logits = torch.softmax(output, dim=-1)
          pred  = torch.argmax(logits , dim=-1)
          loss= self.criterion(output, y.long()) if y is not None else 0
        else:
          logits = torch.sigmoid(output).float()
          pred =  (logits>0.5).float()
          loss= self.criterion(logits, y.float()) if y is not None else 0

        result = {'logits': logits, 'loss': loss, 'pred':pred}

        return result


    # lightning training step
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.mode == "four":
          forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch["token_type_ids"], batch['sentiment_four'] )
        if self.mode == "bin":
          forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch["token_type_ids"], batch['sentiment_bin'] )

        return forward_output['loss']


    # lightning validation step
    def validation_step(self,  batch: Tuple[torch.Tensor],  batch_idx: int):
        if self.mode=="four":
          forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch["token_type_ids"] , batch['sentiment_four'] )
          f1 = self.val_f1(forward_output['pred'].view(-1), batch['sentiment_four'].view(-1))
        else:
          forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch["token_type_ids"], batch['sentiment_bin'] )
          y_hat = []; y = []
          for i, x in enumerate(forward_output['logits']):
            y_hat += [sent2idx[pt2sent(x)]]
            y += [sent2idx[pt2sent(batch['sentiment_bin'][i])]]
          f1 = self.val_f1(torch.tensor(y_hat).view(-1).cuda(), torch.tensor(y).view(-1).cuda())

        self.log('val_f1', f1, prog_bar=True)
        self.log('val_loss', forward_output['loss'], prog_bar=True)


    def test_step(self,batch: Tuple[torch.Tensor],batch_idx: int):
        pass


    # return the optimizer
    def configure_optimizers(self):
        return AdamW(self.parameters(), self.train_param['lr'], eps=self.train_param['eps'])



class Classifier_C(pl.LightningModule):
    """
    A model for aspect based sentiment analysis (TASK C) based on bert + linear
    Model for multi-label classification
    ...

    Attributes
    ----------
    train_param : dict
        a dict were all the training paramter as learning rate or loss weight are provided.
    bert : torch.nn.module
        pretrained bert base cased from hugginface
    dropout : torch.nn.dropout 
        dropout between bert and lstm
    classifier : torch.nn.module
        the final linear layer of the network
    criterion : torch.nn.CrossEntropyLoss
        loss of the network
    val_f1 : torchmetrics
        provide macro f1 score during training
    """
    
    # constructor
    def __init__(self, train_param:Dict, path="./model/pre_trained"):
        super().__init__()
        
        self.train_param = train_param
        self.bert = BertModel.from_pretrained(path)
        self.dropout = nn.Dropout(p=self.train_param['dropout'])
        self.classifier = nn.Linear(self.bert.config.hidden_size,5)
        self.criterion = nn.BCELoss()

        # freeze bert weight --> use it as a feature extrator
        if self.train_param['freeze']:
          for param in self.bert.parameters():
            param.requires_grad = False


    # single forward step 
    def forward(self, x: torch.Tensor, mask: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        output = self.bert(input_ids=x, attention_mask=mask)
        output = self.classifier(self.dropout(output.pooler_output))
        logits = torch.sigmoid(output)

        loss = 0
        if y is not None:
            loss = self.criterion(logits.float(), y.float())
        
        pred =  (logits>0.5).float()

        result = {'logits': logits, 'loss': loss, 'pred':pred}
        return result


    # lightning training step
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch['categories_label'] )
        return forward_output['loss']


    # lightning validation step
    def validation_step(self,  batch: Tuple[torch.Tensor],  batch_idx: int):
        forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch['categories_label'] )

        out_aspect = []
        y_aspect = []
        for i, x in enumerate(forward_output['pred']):
            out = []
            for a, c in enumerate(x):
              if c.item() == 1:
                out +=[ idx2cat[a] ]
            out_aspect += [out]
            y_aspect += [ [x[0] for x in batch['categories'][i]] ]

        metric = evaluate_extraction( y_aspect, out_aspect)
        self.log('val_f1', metric['f1'], prog_bar=True)
        self.log('val_loss', forward_output['loss'], prog_bar=True)


    def test_step(self,batch: Tuple[torch.Tensor],batch_idx: int):
        forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch['sentiment'] )
        self.log('test_f1', self.val_f1, prog_bar=True)


    # return the optimizer
    def configure_optimizers(self):
        return AdamW(self.bert.parameters(), self.train_param['lr'], eps=self.train_param['eps'])



class Classifier_D(pl.LightningModule):
    """
    A model for aspect based sentiment analysis (TASK D) based on bert + linear
    ...

    Attributes
    ----------
    train_param : dict
        a dict were all the training paramter as learning rate or loss weight are provided.
    mode : str
        'four' or 'bin' to select final output layer (see report)
    bert : torch.nn.module
        pretrained bert base cased from hugginface
    dropout : torch.nn.dropout 
        dropout between bert and lstm
    classifier : torch.nn.module
        the final linear layer of the network
    criterion : torch.nn.CrossEntropyLoss
        loss of the network
    val_f1 : torchmetrics
        provide macro f1 score during training
    """
    
    # constructor
    def __init__(self, train_param:Dict, mode = "four", path="./model/pre_trained" ):
        super().__init__()

        self.train_param = train_param
        self.mode = mode
        self.bert = BertModel.from_pretrained(path)
        self.dropout = nn.Dropout(p=self.train_param['dropout'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, 4 if self.mode=="four" else 2)
        self.criterion = nn.CrossEntropyLoss(weight=self.train_param['loss_weight']) if self.mode=="four" else nn.BCELoss()
        self.val_f1 =  tm.F1(num_classes=4,average='macro')
 
        # freeze bert weight --> use it as a feature extrator 
        if self.train_param['freeze']:
          for param in self.bert.parameters():
            param.requires_grad = False


    # single forward step 
    def forward(self, x: torch.Tensor, mask: torch.Tensor, token_type: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        output = self.bert(input_ids=x, attention_mask=mask, token_type_ids= token_type)
        output = self.classifier(self.dropout(output.pooler_output))

        if self.mode == "four":
          logits = torch.softmax(output, dim=-1)
          pred  = torch.argmax(logits , dim=-1)
          loss= self.criterion(output, y.long()) if y is not None else 0
          
        else:
          logits = torch.sigmoid(output).float()
          pred =  (logits>0.5).float()
          loss= self.criterion(logits, y.float()) if y is not None else 0
          
        result = {'logits': logits, 'loss': loss, 'pred':pred}

        return result


    # lightning training step
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.mode == "four":
          forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch["token_type_ids"], batch['sentiment_four'] )
        if self.mode == "bin":
          forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch["token_type_ids"], batch['sentiment_bin'] )

        return forward_output['loss']


    # lightning validation step
    def validation_step(self,  batch: Tuple[torch.Tensor],  batch_idx: int):
    
        if self.mode=="four":
          forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch["token_type_ids"], batch['sentiment_four'] )
          f1 = self.val_f1(forward_output['pred'].view(-1), batch['sentiment_four'].view(-1))
          
        else:
          forward_output = self.forward(batch['input_ids'], batch['attention_mask'] , batch["token_type_ids"], batch['sentiment_bin'] )
          y_hat = []; y = []
          for i, x in enumerate(forward_output['logits']):
            y_hat += [sent2idx[pt2sent(x)]]
            y += [sent2idx[pt2sent(batch['sentiment_bin'][i])]]
          f1 = self.val_f1(torch.tensor(y_hat).view(-1).cuda(), torch.tensor(y).view(-1).cuda())

        self.log('val_f1', f1, prog_bar=True)
        self.log('val_loss', forward_output['loss'], prog_bar=True)


    def test_step(self,batch: Tuple[torch.Tensor],batch_idx: int):
        pass


    # return the optimizer
    def configure_optimizers(self):
        return AdamW(self.parameters(), self.train_param['lr'], eps=self.train_param['eps'])

