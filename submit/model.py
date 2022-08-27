import transformers as trs
import torch
import torch.nn as nn
import numpy as np

'''
USAGE: This file is for INFERENCE ONLY, TRAINING will be done in TRAIN.py
ONLY use this AFTER TRAINING and EXPORTING WEIGHTS is COMPLETE
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QHD_Model(nn.Module):
    def __init__(self, seperate) -> None:
        super(QHD_Model, self).__init__()
    
        self.mode = seperate 
        
        if not seperate:
            self.kernels = nn.Sequential(*[
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(768, 6),
                nn.Sigmoid()
            ])
        
        else:
            self.kernels = []
            self.kernels.extend(
                [nn.Sequential(*[
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 6),
                nn.Sigmoid()]) for i in range(6)]
            )
    
    def forward(self, x):
        if not self.mode:
            out = self.kernels(x)
        else:
            out = self.kernels[0](x)
            for i in range(1, len(self.kernels)):
                out = torch.concat((out, self.kernels[i](x)), axis = 1)
        
        return out

class ReviewClassifierModel(nn.Module):
    def __init__(self):
        super(ReviewClassifierModel, self).__init__()
        self.tokenizer = trs.AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert = trs.AutoModel.from_pretrained("vinai/phobert-base")
        self.classifier = QHD_Model(False)

    def setup_classifier(self):
        #TEMPORARY, WILL ADD CONFIG TO CLASSIFIER LATER
        self.classifier.load_state_dict(torch.load("ckpt_model"))

    def __call__(self, text):
        #Single input
        out = self.tokenizer.encode(text, truncation=True, padding=True,  return_tensors="pt")
        out = self.phobert(out)
        with torch.no_grad():
            out = self.classifier(out[1]).cpu().detach().numpy()
        #HAVEN'T HANDLE YET
            out = np.reshape(out, (6))

        print(out[5])
        return out




