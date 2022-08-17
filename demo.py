import processing
import model

if __name__ == '__main__':
    processor = processing.TextProcessing("vinai/phobert-base")
    processor.read_csv("data.csv")
    train_dataset, val_dataset = processor.process()

    phobert = model.BERT("vinai/phobert-base")
    phobert.train(train_dataset, val_dataset)
