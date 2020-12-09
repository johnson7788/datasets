from pyarrow import fs

local = fs.LocalFileSystem()
# info = local.get_file_info(fs.FileSelector("weibo"))
# print(info)
# info1 = local.get_file_info(fs.FileSelector("weibo/", recursive=True))
# print(info1)

from datasets.arrow_reader import ArrowReader
from datasets import DatasetInfo

arrow_dir = "weibo/"
info_dir = "weibo/"
#读取dataset_info.json文件，转换成dict格式
info = DatasetInfo.from_directory(info_dir)
#初始化一个arrow reader
myreader = ArrowReader(path=arrow_dir,info=info)
#读取目录下的weibo_ner_corpus-train.arrow文件，跳过skip行，读取到take行，这里是读取前100条
files = [{'filename': 'weibo_ner_corpus-train.arrow', 'skip': 0, 'take': 100}]
#这里instructions只是标记用，暂时
instructions = "train"
# 读取文件，生成arrow格式的专用table，类似padas的DataFrame格式
dataset_kwargs = myreader.read_files(files=files, original_instructions=instructions)
#打印第2列的内容
print(dataset_kwargs['arrow_table'].column(2))


#read_files
"""
read_files(files=files, original_instructions=instructions)
files = {list: 1} [{'filename': 'msra_ner-train.arrow', 'skip': 0, 'take': 45001}]
instructions = {NamedSplit} train
"""

#read方法
"""
instructions = {NamedSplit} train
name = {str} 'msra_ner'
self = {ArrowReader} <datasets.arrow_reader.ArrowReader object at 0x7fb60a40a310>
split_infos = {dict_values: 2} dict_values([SplitInfo(name='train', num_bytes=33323074, num_examples=45001, dataset_name='msra_ner'), SplitInfo(name='test', num_bytes=2642934, num_examples=3443, dataset_name='msra_ner')])
"""