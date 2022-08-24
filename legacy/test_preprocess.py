from legacy import preprocess

if __name__ == '__main__':
    processor = preprocess.TFTextProcessing()
    processor.process("data.csv")
