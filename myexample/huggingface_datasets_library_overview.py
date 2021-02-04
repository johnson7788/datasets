# -*- coding: utf-8 -*-
"""HuggingFace datasets library - Overview

Original file is located at
    https://colab.research.google.com/github/huggingface/datasets/blob/master/notebooks/Overview.ipynb

该库具有几个有趣的特征(易于访问数据集/指标)：

-与PyTorch，Tensorflow 2，Pandas和Numpy的内置互操作性
-轻量级且具有透明特征的快速库 和pythonic API
-努力使用大型数据集：使您摆脱RAM内存限制，默认情况下所有数据集都映射到驱动器上的内存。
-具有类似tf.data`的智能缓存的智能缓存： 永不等待您的数据处理多次

“🤗Datasets”源自于很棒的Tensorflow-Datasets的分支，而HuggingFace团队想对这个令人惊叹的库和用户API背后的团队深表感谢。
我们试图与tfds保持兼容性，并且转换可以提供从一种格式到另一种格式的转换。

"""
# pip install datasets


#让我们导入库。 我们通常最多只需要四种方法：
from datasets import list_datasets, list_metrics, load_dataset, load_metric

from pprint import pprint

# 当前可用的数据集和指标
datasets = list_datasets()
metrics = list_metrics()

print(f"🤩 Currently {len(datasets)} datasets are available on the hub:")
pprint(datasets, compact=True)
print(f"🤩 Currently {len(metrics)} metrics are available on the hub:")
pprint(metrics, compact=True)

#您可以在下载数据集之前访问它们的各种属性
squad_dataset = list_datasets(with_details=True)[datasets.index('squad')]

pprint(squad_dataset.__dict__)  # It's a simple python dataclass

# SQuAD的样本

#下载和加载数据集
dataset = load_dataset('squad', split='validation[:10%]')
"""
1.从HuggingFace AWS存储桶中下载并导入** SQuAD python处理脚本**(如果尚未存储在库中)。例如，您可以在[此处](https://github.com/huggingface/datasets/tree/master/datasets/squad/squad.py)中找到SQuAD处理脚本。

处理脚本很小python脚本，用于定义数据集的信息(引用，描述)和格式，并包含原始SQuAD JSON文件的URL和用于从原始SQuAD JSON文件加载样本的代码。

*** * 2。运行SQuAD python处理脚本，该脚本将：
-**如果尚未下载和缓存SQuAD数据集**，请从原始URL下载(请参见脚本)。
-**处理和缓存* *存储在驱动器上的每个标准拆分的结构化箭头表中的所有SQuAD。

ArrowTable是任意长的表，其类型可以映射为numpy / pandas / python标准类型并可以存储嵌套的对象。可以从驱动器直接访问它们，将它们加载到RAM中，甚至可以通过Web流访问它们。

3。从用户要求的分割返回数据集构建(默认值：全部)，在上面的样本中，我们使用验集前10％创建了一个数据集。 
"""

# Informations on the dataset (description, citation, size, splits, format...)
# are provided in `dataset.info` (a simple python dataclass) and also as direct attributes in the dataset object
pprint(dataset.info.__dict__)

"""## Inspecting and using the dataset: elements, slices and columns

The returned `Dataset` object is a memory mapped dataset that behave similarly to a normal map-style dataset. It is backed by an Apache Arrow table which allows many interesting features.
"""

print(dataset)

# “”“您可以查询它的长度，并像使用python映射那样正常获取item或切片。”“”

print(f"👉Dataset len(dataset): {len(dataset)}")
print("\n👉First item 'dataset[0]':")
pprint(dataset[0])

# Or get slices with several examples:
print("\n👉Slice of the two items 'dataset[10:12]':")
pprint(dataset[10:12])

# You can get a full column of the dataset by indexing with its name as a string:
print(dataset['question'][:10])

"""The `__getitem__` method will return different format depending on the type of query:

- Items like `dataset[0]` are returned as dict of elements.
- Slices like `dataset[10:20]` are returned as dict of lists of elements.
- Columns like `dataset['question']` are returned as a list of elements.

This may seems surprising at first but in our experiments it's actually a lot easier to use for data processing than returning the same format for each of these views on the dataset.

In particular, you can easily iterate along columns in slices, and also naturally permute consecutive indexings with identical results as showed here by permuting column indexing with elements and slices:
"""

print(dataset[0]['question'] == dataset['question'][0])
print(dataset[10:20]['context'] == dataset['context'][10:20])

"""### Dataset are internally typed and structured

The dataset is backed by one (or several) Apache Arrow tables which are typed and allows for fast retrieval and access as well as arbitrary-size memory mapping.

This means respectively that the format for the dataset is clearly defined and that you can load datasets of arbitrary size without worrying about RAM memory limitation (basically the dataset take no space in RAM, it's directly read from drive when needed with fast IO access).
"""

# You can inspect the dataset column names and types 
print("Column names:")
pprint(dataset.column_names)
print("Features:")
pprint(dataset.features)

"""### Additional misc properties"""

# Datasets also have shapes informations
print("The number of rows", dataset.num_rows, "also available as len(dataset)", len(dataset))
print("The number of columns", dataset.num_columns)
print("The shape (rows, columns)", dataset.shape)

"""## Modifying the dataset with `dataset.map`

Now that we know how to inspect our dataset we also want to update it. For that there is a powerful method `.map()` which is inspired by `tf.data` map method and that you can use to apply a function to each examples, independently or in batch.

`.map()` takes a callable accepting a dict as argument (same dict as the one returned by `dataset[i]`) and iterate over the dataset by calling the function on each example.
"""

# Let's print the length of each `context` string in our subset of the dataset
# (10% of the validation i.e. 1057 examples)

dataset.map(lambda example: print(len(example['context']), end=','))

"""This is basically the same as doing

```python
for example in dataset:
    function(example)
```

上面的样本有些冗长。 我们可以使用它的日志记录模块来控制`🤗Datasets`的日志记录级别：
"""

from datasets import logging
logging.set_verbosity_warning()

dataset.map(lambda example: print(len(example['context']), end=','))

# Let's keep it verbose for our tutorial though
from datasets import logging
logging.set_verbosity_info()

"""The above example had no effect on the dataset because the method we supplied to `.map()` didn't return a `dict` or a `abc.Mapping` that could be used to update the examples in the dataset.

In such a case, `.map()` will return the same dataset (`self`).

Now let's see how we can use a method that actually modify the dataset.

### Modifying the dataset example by example

The main interest of `.map()` is to update and modify the content of the table and leverage smart caching and fast backend.

To use `.map()` to update elements in the table you need to provide a function with the following signature: `function(example: dict) -> dict`.
"""

# Let's add a prefix 'My cute title: ' to each of our titles

def add_prefix_to_title(example):
    example['title'] = 'My cute title: ' + example['title']
    return example

prefixed_dataset = dataset.map(add_prefix_to_title)

print(prefixed_dataset.unique('title'))  # `.unique()` is a super fast way to print the unique elemnts in a column (see the doc for all the methods)

"""This call to `.map()` compute and return the updated table. It will also store the updated table in a cache file indexed by the current state and the mapped function.

A subsequent call to `.map()` (even in another python session) will reuse the cached file instead of recomputing the operation.

You can test this by running again the previous cell, you will see that the result are directly loaded from the cache and not re-computed again.

The updated dataset returned by `.map()` is (again) directly memory mapped from drive and not allocated in RAM.

The function you provide to `.map()` should accept an input with the format of an item of the dataset: `function(dataset[0])` and return a python dict.

The columns and type of the outputs can be different than the input dict. In this case the new keys will be added as additional columns in the dataset.

Bascially each dataset example dict is updated with the dictionary returned by the function like this: `example.update(function(example))`.
"""

# Since the input example dict is updated with our function output dict,
# we can actually just return the updated 'title' field
titled_dataset = dataset.map(lambda example: {'title': 'My cutest title: ' + example['title']})

print(titled_dataset.unique('title'))

"""#### Removing columns
You can also remove columns when running map with the `remove_columns=List[str]` argument.
"""

# This will remove the 'title' column while doing the update (after having send it the the mapped function so you can use it in your function!)
less_columns_dataset = dataset.map(lambda example: {'new_title': 'Wouhahh: ' + example['title']}, remove_columns=['title'])

print(less_columns_dataset.column_names)
print(less_columns_dataset.unique('new_title'))

"""#### Using examples indices
With `with_indices=True`, dataset indices (from `0` to `len(dataset)`) will be supplied to the function which must thus have the following signature: `function(example: dict, indice: int) -> dict`
"""

# This will add the index in the dataset to the 'question' field
with_indices_dataset = dataset.map(lambda example, idx: {'question': f'{idx}: ' + example['question']},
                                   with_indices=True)

pprint(with_indices_dataset['question'][:5])

"""### Modifying the dataset with batched updates

`.map()` can also work with batch of examples (slices of the dataset).

This is particularly interesting if you have a function that can handle batch of inputs like the tokenizers of HuggingFace `tokenizers`.

To work on batched inputs set `batched=True` when calling `.map()` and supply a function with the following signature: `function(examples: Dict[List]) -> Dict[List]` or, if you use indices, `function(examples: Dict[List], indices: List[int]) -> Dict[List]`).

Bascially, your function should accept an input with the format of a slice of the dataset: `function(dataset[:10])`.
"""


# Let's import a fast tokenizer that can work on batched inputs
# (the 'Fast' tokenizers in HuggingFace)
from transformers import BertTokenizerFast, logging as transformers_logging

transformers_logging.set_verbosity_warning()

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# Now let's batch tokenize our dataset 'context'
encoded_dataset = dataset.map(lambda example: tokenizer(example['context']), batched=True)

print("encoded_dataset[0]")
pprint(encoded_dataset[0], compact=True)

# we have added additional columns
pprint(dataset.column_names)

# Let show a more complex processing with the full preparation of the SQuAD dataset
# for training a model from Transformers
def convert_to_features(batch):
    # Tokenize contexts and questions (as pairs of inputs)
    input_pairs = list(zip())
    encodings = tokenizer(batch['context'], batch['question'], truncation=True)

    # Compute start and end tokens for labels
    start_positions, end_positions = [], []
    for i, answer in enumerate(batch['answers']):
        first_char = answer['answer_start'][0]
        last_char = first_char + len(answer['text'][0]) - 1
        start_positions.append(encodings.char_to_token(i, first_char))
        end_positions.append(encodings.char_to_token(i, last_char))

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    return encodings

encoded_dataset = dataset.map(convert_to_features, batched=True)

# Now our dataset comprise the labels for the start and end position
# as well as the offsets for converting back tokens
# in span of the original string for evaluation
print("column_names", encoded_dataset.column_names)
print("start_positions", encoded_dataset[:5]['start_positions'])

"""## formatting outputs for PyTorch, Tensorflow, Numpy, Pandas

Now that we have tokenized our inputs, we probably want to use this dataset in a `torch.Dataloader` or a `tf.data.Dataset`.

To be able to do this we need to tweak two things:

- format the indexing (`__getitem__`) to return numpy/pytorch/tensorflow tensors, instead of python objects, and probably
- format the indexing (`__getitem__`) to return only the subset of the columns that we need for our model inputs.

  We don't want the columns `id` or `title` as inputs to train our model, but we could still want to keep them in the dataset, for instance for the evaluation of the model.
    
This is handled by the `.set_format(type: Union[None, str], columns: Union[None, str, List[str]])` where:

- `type` define the return type for our dataset `__getitem__` method and is one of `[None, 'numpy', 'pandas', 'torch', 'tensorflow']` (`None` means return python objects), and
- `columns` define the columns returned by `__getitem__` and takes the name of a column in the dataset or a list of columns to return (`None` means return all columns).
"""

columns_to_return = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']

encoded_dataset.set_format(type='torch', columns=columns_to_return)

# Our dataset indexing output is now ready for being used in a pytorch dataloader
pprint(encoded_dataset[1], compact=True)

# Note that the columns are not removed from the dataset, just not returned when calling __getitem__
# Similarly the inner type of the dataset is not changed to torch.Tensor, the conversion and filtering is done on-the-fly when querying the dataset
print(encoded_dataset.column_names)

# We can remove the formatting with `.reset_format()`
# or, identically, a call to `.set_format()` with no arguments
encoded_dataset.reset_format()

pprint(encoded_dataset[1], compact=True)

# The current format can be checked with `.format`,
# which is a dict of the type and formatting
pprint(encoded_dataset.format)

"""# Wrapping this all up (PyTorch)

Let's wrap this all up with the full code to load and prepare SQuAD for training a PyTorch model from HuggingFace `transformers` library.


"""


import torch 
from datasets import load_dataset
from transformers import BertTokenizerFast

# Load our training dataset and tokenizer
dataset = load_dataset('squad')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

def get_correct_alignement(context, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text'][0]
    start_idx = answer['answer_start'][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       # When the gold label position is good
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1   # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2   # When the gold label is off by two character
    else:
        raise ValueError()

# Tokenize our training dataset
def convert_to_features(example_batch):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer(example_batch['context'], example_batch['question'], truncation=True)

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methods.
    start_positions, end_positions = [], []
    for i, (context, answer) in enumerate(zip(example_batch['context'], example_batch['answers'])):
        start_idx, end_idx = get_correct_alignement(context, answer)
        start_positions.append(encodings.char_to_token(i, start_idx))
        end_positions.append(encodings.char_to_token(i, end_idx-1))
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    return encodings

encoded_dataset = dataset.map(convert_to_features, batched=True)

# Format our dataset to outputs torch.Tensor to train a pytorch model
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
encoded_dataset.set_format(type='torch', columns=columns)

# Instantiate a PyTorch Dataloader around our dataset
# Let's do dynamic batching (pad on the fly with our own collate_fn)
def collate_fn(examples):
    return tokenizer.pad(examples, return_tensors='pt')
dataloader = torch.utils.data.DataLoader(encoded_dataset['train'], collate_fn=collate_fn, batch_size=8)

# Let's load a pretrained Bert model and a simple optimizer
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('distilbert-base-cased', return_dict=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Now let's train our model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.train().to(device)
for i, batch in enumerate(dataloader):
    batch.to(device)
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    model.zero_grad()
    print(f'Step {i} - loss: {loss:.3}')
    if i > 5:
        break

"""# Wrapping this all up (Tensorflow)

Let's wrap this all up with the full code to load and prepare SQuAD for training a Tensorflow model (works only from the version 2.2.0)
"""

import tensorflow as tf
import datasets
from transformers import BertTokenizerFast

# Load our training dataset and tokenizer
train_tf_dataset = datasets.load_dataset('squad', split="train")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', return_dict=True)

# Tokenize our training dataset
# The only one diff here is that start_positions and end_positions
# must be single dim list => [[23], [45] ...]
# instead of => [23, 45 ...]
def convert_to_tf_features(example_batch):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer(example_batch['context'], example_batch['question'], truncation=True)

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methods.
    start_positions, end_positions = [], []
    for i, (context, answer) in enumerate(zip(example_batch['context'], example_batch['answers'])):
        start_idx, end_idx = get_correct_alignement(context, answer)
        start_positions.append([encodings.char_to_token(i, start_idx)])
        end_positions.append([encodings.char_to_token(i, end_idx-1)])
    
    if start_positions and end_positions:
      encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    return encodings

train_tf_dataset = train_tf_dataset.map(convert_to_tf_features, batched=True)

def remove_none_values(example):
  return not None in example["start_positions"] or not None in example["end_positions"]

train_tf_dataset = train_tf_dataset.filter(remove_none_values, load_from_cache_file=False)
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
train_tf_dataset.set_format(type='tensorflow', columns=columns)
features = {x: train_tf_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.max_len]) for x in columns[:3]} 
labels = {"output_1": train_tf_dataset["start_positions"].to_tensor(default_value=0, shape=[None, 1])}
labels["output_2"] = train_tf_dataset["end_positions"].to_tensor(default_value=0, shape=[None, 1])
tfdataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(8)

# Let's load a pretrained TF2 Bert model and a simple optimizer
from transformers import TFBertForQuestionAnswering

model = TFBertForQuestionAnswering.from_pretrained("bert-base-cased")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=opt,
              loss={'output_1': loss_fn, 'output_2': loss_fn},
              loss_weights={'output_1': 1., 'output_2': 1.},
              metrics=['accuracy'])

# Now let's train our model

model.fit(tfdataset, epochs=1, steps_per_epoch=3)

"""# Metrics API

`datasets` also provides easy access and sharing of metrics.

This aspect of the library is still experimental and the API may still evolve more than the datasets API.

Like datasets, metrics are added as small scripts wrapping common metrics in a common API.

There are several reason you may want to use metrics with `datasets` and in particular:

- metrics for specific datasets like GLUE or SQuAD are provided out-of-the-box in a simple, convenient and consistant way integrated with the dataset,
- metrics in `datasets` leverage the powerful backend to provide smart features out-of-the-box like support for distributed evaluation in PyTorch

## Using metrics

Using metrics is pretty simple, they have two main methods: `.compute(predictions, references)` to directly compute the metric and `.add(prediction, reference)` or `.add_batch(predictions, references)` to only store some results if you want to do the evaluation in one go at the end.

Here is a quick gist of a standard use of metrics (the simplest usage):
```python
from datasets import load_metric
sacrebleu_metric = load_metric('sacrebleu')

# If you only have a single iteration, you can easily compute the score like this
predictions = model(inputs)
score = sacrebleu_metric.compute(predictions, references)

# If you have a loop, you can "add" your predictions and references at each iteration instead of having to save them yourself (the metric object store them efficiently for you)
for batch in dataloader:
    model_input, targets = batch
    predictions = model(model_inputs)
    sacrebleu_metric.add_batch(predictions, targets)
score = sacrebleu_metric.compute()  # Compute the score from all the stored predictions/references
```

Here is a quick gist of a use in a distributed torch setup (should work for any python multi-process setup actually). It's pretty much identical to the second example above:
```python
from datasets import load_metric
# You need to give the total number of parallel python processes (num_process) and the id of each process (process_id)
bleu_metric = datasets.load_metric('sacrebleu', process_id=torch.distributed.get_rank(),b num_process=torch.distributed.get_world_size())

for batch in dataloader:
    model_input, targets = batch
    predictions = model(model_inputs)
    sacrebleu_metric.add_batch(predictions, targets)
score = sacrebleu_metric.compute()  # Compute the score on the first node by default (can be set to compute on each node as well)
```

Example with a NER metric: `seqeval`
"""

ner_metric = load_metric('seqeval')
references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
predictions =  [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
ner_metric.compute(predictions, references)

"""# Adding a new dataset or a new metric

They are two ways to add new datasets and metrics in `datasets`:

- datasets can be added with a Pull-Request adding a script in the `datasets` folder of the [`datasets` repository](https://github.com/huggingface/datasets)

=> once the PR is merged, the dataset can be instantiate by it's folder name e.g. `datasets.load_dataset('squad')`. If you want HuggingFace to host the data as well you will need to ask the HuggingFace team to upload the data.

- datasets can also be added with a direct upload using `datasets` CLI as a user or organization (like for models in `transformers`). In this case the dataset will be accessible under the gien user/organization name, e.g. `datasets.load_dataset('thomwolf/squad')`. In this case you can upload the data yourself at the same time and in the same folder.

We will add a full tutorial on how to add and upload datasets soon.
"""

