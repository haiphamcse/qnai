from pkgutil import get_data
import torch.nn as nn
import torch
import transformers as trs
from model import *
from dataset import *
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

def balance_loss(y, y_pred, alpha = 0.9, beta = 0.1):
    loss =  - (1.-y_pred)**alpha * y * torch.log(y_pred) -  (y_pred) ** beta * (1.-y)* torch.log(1. - y_pred)
    loss = torch.sum(loss, dim = 1) / 6.0
    loss = torch.mean(loss)
    return loss.to(device)

def mse(y, y_pred):
    square = (y - y_pred) ** 2
    return torch.mean(square)

def train(model: nn.Module, mode_train, epochs: int, batch_size: int, lr: float, train_dataset, val_dataset, save_path_model) -> None:
    
    if not model.mode:  #   Not seperate
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    else:
        optimizer = torch.optim.Adam(model.kernels.parameters(), lr = lr)
    
    train_losses = [1.5]
    val_losses = [1.5]
    model.train()

    train_loader = get_data_loader(dataset=train_dataset, batch_size=batch_size, shuffle = None)
    val_loader = get_data_loader(dataset=val_dataset, batch_size = batch_size, shuffle = None)

    tokenizer = trs.AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast = True)
    phobert = trs.AutoModel.from_pretrained("vinai/phobert-base")

    for param in phobert.parameters():
        param.requires_grad = False

    #   Tokenize text -> Embedding
    for epoch in range(epochs):

        for (idx, batch) in tqdm(enumerate(train_loader)):
            text, label = batch
            text = list(text)
            #   Label for training classifier
            if mode_train == "classify":
                label = (label > 0).type(torch.float64).to(device)
            else:
                label = label.type(torch.float64).to(device)
            
            #   Clean and text
            with torch.no_grad():
                feature = tokenizer(text, truncation = True, padding = True, return_tensors = "pt")                
                feature = phobert(input_ids = feature['input_ids'], token_type_ids = feature['token_type_ids'])
                
            optimizer.zero_grad()
            logits = model(feature[1])

            if mode_train == "classify":
                loss = balance_loss(label, logits)
            elif mode_train == "regression":
                loss = mse(label, logits)
            loss.backward()
            optimizer.step()
        
        current_train_loss = loss.item()
        
        print("Testing on validation set...")
        #   Test on validation dataset
        for(idx, val_batch) in tqdm(enumerate(val_loader)):
            val_text, val_label = val_batch
            if mode_train == "classify":
                val_label = (val_label > 0).type(torch.float64).to(device)
            else:
                val_label = val_label.type(torch.float64).to(device)
            val_text = list(val_text)

            with torch.no_grad():
                val_feature = tokenizer(val_text, truncation = True, padding = True, return_tensors = "pt")
                val_feature = phobert(input_ids = val_feature['input_ids'], token_type_ids = val_feature['token_type_ids'])
                val_logits = model(val_feature[1])
                if mode_train == "classify":
                    val_loss = balance_loss(val_label, val_logits)
                else :
                    val_loss = mse(val_logits, val_label)

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
    classifier_model = QHD_Model(False).to(device)
    regressor_model = QHD_Regressor(False).to(device)

    epochs = 50
    lr = 1e-2
    batch_size = 150
    mode_train = "regression"

    train_dataset = QNAIDataset("data.csv", "train")
    val_dataset = QNAIDataset("data.csv", "test")
    parse = argparse.ArgumentParser()    
    parse.add_argument("--save_model", required=False)
    parser = parse.parse_args()

    save_model_path = parser.save_model
    train_loss, val_loss = train(regressor_model, mode_train, epochs, batch_size, lr, train_dataset, val_dataset, save_model_path)