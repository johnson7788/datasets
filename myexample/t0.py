from datasets import list_datasets, load_dataset, list_metrics, load_metric

msra_dataset = load_dataset('/Users/admin/git/datasets/datasets/msra_ner/msra_ner.py')
print(msra_dataset['train'][:5])