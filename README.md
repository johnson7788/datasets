<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/datasets/master/docs/source/imgs/datasets_logo_name.jpg" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/datasets">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/datasets/master">
    </a>
    <a href="https://github.com/huggingface/datasets/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/datasets/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/datasets/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/datasets/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/datasets.svg">
    </a>
    <a href="https://huggingface.co/datasets/">
        <img alt="Number of datasets" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/datasets&color=brightgreen">
    </a>
</p>

`🤗Datasets` is a lightweight library providing **two** main features:

- **one-line dataloaders for many public datasets**: one liners to download and pre-process any of the ![number of datasets](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/datasets&color=brightgreen) major public datasets (in 467 languages and dialects!) provided on the [HuggingFace Datasets Hub](https://huggingface.co/datasets). With a simple command like `squad_dataset = load_datasets("squad")`, get any of these datasets ready to use in a dataloader for training/evaluating a ML model (Numpy/Pandas/PyTorch/TensorFlow/JAX),
- **efficient data pre-processing**: simple, fast and reproducible data pre-processing for the above public datasets as well as your own local datasets in CSV/JSON/text. With simple commands like `tokenized_dataset = dataset.map(tokenize_exemple)`, efficiently prepare the dataset for inspection and ML model evaluation and training.

[🎓 **Documentation**](https://huggingface.co/docs/datasets/) [🕹 **Colab tutorial**](https://colab.research.google.com/github/huggingface/datasets/blob/master/notebooks/Overview.ipynb)

[🔎 **Find a dataset in the Hub**](https://huggingface.co/datasets) [🌟 **Add a new dataset to the Hub**](https://github.com/huggingface/datasets/blob/master/ADD_NEW_DATASET.md)

<<<<<<< HEAD
🤗Datasets是一个轻量级且可扩展的库，可轻松共享和访问用于自然语言处理(NLP)等的数据集和评估指标。

🤗Datasets具有许多有趣的特征(除了易于共享和 访问数据集/指标)：

-与NumPy，Pandas，PyTorch和Tensorflow 2的内置互操作性
-轻巧且快速的透明和Pythonic API 
-力求大 数据集：“🤗Datasets”自然使用户摆脱了RAM内存限制，默认情况下所有数据集都映射到驱动器上。
-智能缓存：从不等待数据处理几次

🤗Datasets当前可访问约100个NLP数据集和约10个评估指标，旨在让社区轻松添加和共享新的数据集和评估指标。
 您可以使用来浏览完整的数据集  [live datasets viewer](https://huggingface.co/datasets/viewer).

“🤗Datasets”源自令人敬畏的[`TensorFlow Datasets`](https://github.com/tensorflow/datasets)的分支，
而HuggingFace团队要深深感谢TensorFlow Datasets团队构建了这个惊人的库。 
关于[datasets]与`tfds`之间差异的更多详细信息，请参见[[[Datasets`与'tfds`之间的主要差异](#main-differences-between-🤗Datasetsand tfds)。** ** 

# Contributors

[![](https://sourcerer.io/fame/thomwolf/huggingface/datasets/images/0)](https://sourcerer.io/fame/thomwolf/huggingface/datasets/links/0)[![](https://sourcerer.io/fame/thomwolf/huggingface/datasets/images/1)](https://sourcerer.io/fame/thomwolf/huggingface/datasets/links/1)[![](https://sourcerer.io/fame/thomwolf/huggingface/datasets/images/2)](https://sourcerer.io/fame/thomwolf/huggingface/datasets/links/2)[![](https://sourcerer.io/fame/thomwolf/huggingface/datasets/images/3)](https://sourcerer.io/fame/thomwolf/huggingface/datasets/links/3)[![](https://sourcerer.io/fame/thomwolf/huggingface/datasets/images/4)](https://sourcerer.io/fame/thomwolf/huggingface/datasets/links/4)[![](https://sourcerer.io/fame/thomwolf/huggingface/datasets/images/5)](https://sourcerer.io/fame/thomwolf/huggingface/datasets/links/5)[![](https://sourcerer.io/fame/thomwolf/huggingface/datasets/images/6)](https://sourcerer.io/fame/thomwolf/huggingface/datasets/links/6)[![](https://sourcerer.io/fame/thomwolf/huggingface/datasets/images/7)](https://sourcerer.io/fame/thomwolf/huggingface/datasets/links/7)
=======
`🤗Datasets` also provides access to +15 evaluation metrics and is designed to let the community easily add and share new datasets and evaluation metrics. 

`🤗Datasets` has many additional interesting features:
- Thrive on large datasets: `🤗Datasets` naturally frees the user from RAM memory limitation, all datasets are memory-mapped using an efficient zero-serialization cost backend (Apache Arrow).
- Smart caching: never wait for your data to process several times.
- Lightweight and fast with a transparent and pythonic API (multi-processing/caching/memory-mapping).
- Built-in interoperability with NumPy, pandas, PyTorch, Tensorflow 2 and JAX.

`🤗Datasets` originated from a fork of the awesome [`TensorFlow Datasets`](https://github.com/tensorflow/datasets) and the HuggingFace team want to deeply thank the TensorFlow Datasets team for building this amazing library. More details on the differences between `🤗Datasets` and `tfds` can be found in the section [Main differences between `🤗Datasets` and `tfds`](#main-differences-between-datasets-and-tfds).
>>>>>>> upstream/master

# Installation

🤗Datasets可以从PyPi安装，必须安装在虚拟环境中(例如，venv或conda)

```bash
pip install datasets
```

有关安装的更多详细信息，请查看文档中的安装页面。 : https://huggingface.co/docs/datasets/installation.html

## Installation to use with PyTorch/TensorFlow/pandas

如果您打算在PyTorch(1.0 +)，TensorFlow(2.2+)或pandas中使用`🤗Datasets`，则还应该安装PyTorch，TensorFlow或pandas。

有关将库与NumPy，pandas，PyTorch或TensorFlow一起使用的更多详细信息，请查看文档中的快速浏览页面：https://huggingface.co/docs/datasets/quicktour.html

# Usage

🤗Datasets的使用非常简单。 主要方法是：

- `datasets.list_datasets()` 列出可用的数据集 
- `datasets.load_dataset(dataset_name, **kwargs)` 实例化数据集 
- `datasets.list_metrics()` 列出可用指标 
- `datasets.load_metric(metric_name, **kwargs)` 实例化指标 

这是一个简单的样本： 

```python
from datasets import list_datasets, load_dataset, list_metrics, load_metric

#打印所有可用的数据集 
print(list_datasets())

#加载数据集并打印训练集中的第一个样本 
squad_dataset = load_dataset('squad')
print(squad_dataset['train'][0])

#列出所有可用指标 
print(list_metrics())

#加载指标 
squad_metric = load_metric('squad')

# Process the dataset - add a column with the length of the context texts
dataset_with_length = squad_dataset.map(lambda x: {"length": len(x["context"])})

# Process the dataset - tokenize the context texts (using a tokenizer from the 🤗 transformers library)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

tokenized_dataset = squad_dataset.map(lambda x: tokenizer(x['context']), batched=True)
```

有关使用库的更多详细信息，请查看文档中的快速浏览页面。: https://huggingface.co/docs/datasets/quicktour.html 
and the specific pages on
- Loading a dataset https://huggingface.co/docs/datasets/loading_datasets.html
- What's in a Dataset: https://huggingface.co/docs/datasets/exploring.html
- Processing data with `🤗Datasets`: https://huggingface.co/docs/datasets/processing.html
- Writing your own dataset loading script: https://huggingface.co/docs/datasets/add_dataset.html
- etc

Another introduction to `🤗Datasets` is the tutorial on Google Colab here:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/datasets/blob/master/notebooks/Overview.ipynb)

# Add a new dataset to the Hub

We know have a very detailed step-by-step guide to add a new dataset to the ![number of datasets](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/datasets&color=brightgreen) datasets already provided on the [HuggingFace Datasets Hub](https://huggingface.co/datasets).

You will find [the step-by-step guide here](https://github.com/huggingface/datasets/blob/master/ADD_NEW_DATASET.md)

# Main differences between `🤗Datasets` and `tfds`

If you are familiar with the great `Tensorflow Datasets`, here are the main differences between `🤗Datasets` and `tfds`:
- the scripts in `🤗Datasets` are not provided within the library but are queried, downloaded/cached and dynamically loaded upon request
- `🤗Datasets` also provides evaluation metrics in a similar fashion to the datasets, i.e. as dynamically installed scripts with a unified API. This gives access to the pair of a benchmark dataset and a benchmark metric for instance for benchmarks like [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) or [GLUE](https://gluebenchmark.com/).
- the backend serialization of `🤗Datasets` is based on [Apache Arrow](https://arrow.apache.org/) instead of TF Records and leverage python dataclasses for info and features with some diverging features (we mostly don't do encoding and store the raw data as much as possible in the backend serialization cache).
- the user-facing dataset object of `🤗Datasets` is not a `tf.data.Dataset` but a built-in framework-agnostic dataset class with methods inspired by what we like in `tf.data` (like a `map()` method). It basically wraps a memory-mapped Arrow table cache.

# Disclaimers

Similar to `TensorFlow Datasets`, `🤗Datasets` is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use them. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/huggingface/datasets/issues/new). Thanks for your contribution to the ML community!

## BibTeX
If you want to cite this framework you can use this:

```bibtex
@article{2020HuggingFace-datasets,
  title={Datasets},
  author={Thomas Wolf and Quentin Lhoest and Patrick von Platen and Yacine Jernite and Mariama Drame and Julien Plu and Julien Chaumond and Clement Delangue and Clara Ma and Abhishek Thakur and Suraj Patil and Joe Davison and Teven Le Scao and Victor Sanh and Canwen Xu and Nicolas Patry and Angie McMillan-Major and Simon Brandeis and Sylvain Gugger and François Lagunas and Lysandre Debut and Morgan Funtowicz and Anthony Moi and Sasha Rush and Philipp Schmidd and Pierric Cistac and Victor Muštar and Jeff Boudier and Anna Tordjmann},
  journal={GitHub. Note: https://github.com/huggingface/datasets},
  volume={1},
  year={2020}
}
```
