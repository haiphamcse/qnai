from pkgutil import get_data
import torch.nn as nn
import torch
import transformers as trs
from model import *
from dataset import *
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import re
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

def balance_loss(y, y_pred, alpha = 0.9, beta = 0.1):
    loss =  - (1.-y_pred)**alpha * y * torch.log(y_pred) -  (y_pred) ** beta * (1.-y)* torch.log(1. - y_pred)
    loss = torch.sum(loss, dim = 1) / 6.0
    loss = torch.mean(loss)
    return loss.to(device)

def train(model: nn.Module, epochs: int, batch_size: int, lr: float, train_dataset, val_dataset, save_path_model) -> None:
    if not model.mode:  #   Not seperate
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    else:
        optimizer = torch.optim.Adam(model.kernels.parameters(), lr = lr)
    train_losses = [1.5]
    val_losses = [1.5]
    model.train()

    train_loader = get_data_loader(dataset=train_dataset, batch_size=batch_size, shuffle = None)
    val_loader = get_data_loader(dataset=val_dataset, batch_size = batch_size, shuffle = None)

    tokenizer = trs.AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast = False)
    phobert = trs.AutoModel.from_pretrained("vinai/phobert-base")
    

    #   Tokenize text -> Embedding
    for epoch in range(epochs):

        for (idx, batch) in tqdm(enumerate(train_loader)):
            text, label = batch
            label = (label > 0).type(torch.float64)
            
            #   Clean and text
            clean_text = [utils._clean_sentences(sentence) for sentence in text]    
            tokenized_text = [utils._vn_tokenize(sentence) for sentence in clean_text]

            with torch.no_grad():
                feature = tokenizer.encode(tokenized_text, truncation = True, padding = True, return_tensors = "pt")
                feature = phobert(feature)
                feature = feature[1]
                print(feature.shape)
            optimizer.zero_grad()
            logits = model(feature)
            loss = balance_loss(label, logits)
            loss.backward()
            optimizer.step()
        
        current_train_loss = loss.item()

        #   Test on validation dataset
        for(idx, val_batch) in tqdm(enumerate(val_loader)):
            val_text, val_label = val_batch
            val_label = (val_label > 0).type(torch.float64)
            val_text = list(val_text)

            with torch.no_grad():
                val_feature = tokenizer.encode(val_text, truncation = True, padding = True, return_tensors = "pt")
                val_feature = phobert(val_feature)[1]
                val_logits = model(val_feature)
                val_loss = balance_loss(val_label, val_logits)

        current_val_loss = val_loss.item()
        print("Epoch {}: Train loss: {} -- Val loss: {}\n".format(epoch, current_train_loss, current_val_loss))

        if val_loss < min(val_losses):
            print("Val loss decreased!: {} --> {}!".format(min(val_losses), current_val_loss))

            torch.save(model.state_dict(), save_path_model)
            print("Save!\n")
            train_losses.append(current_train_loss)
            val_losses.append(current_val_loss)

    return train_losses, val_losses




if __name__ == '__main__':
    model = QHD_Model(False).to(device)
    epochs = 50
    lr = 1e-2
    batch_size = 150
    train_dataset = QNAIDataset("data.csv", "train")
    val_dataset = QNAIDataset("data.csv", "test")
    parse = argparse.ArgumentParser()    
    parse.add_argument("--save_model", required=True)
    parser = parse.parse_args()

    train_loss, val_loss = train(model, epochs, batch_size, lr, train_dataset, val_dataset, parser.save_model)
    plt.plot(train_loss, val_loss)