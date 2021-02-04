from datasets import load_dataset
dataset = load_dataset('msra/msra_ner.py', data_dir='msra', data_files={'train': 'msra/mini.txt', 'test': 'msra/mini.txt'})
print(dataset)