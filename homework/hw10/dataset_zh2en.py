# From https://data.statmt.org/news-commentary/v15/training/

import os
from torchtext.data.datasets_utils import (
    _download_extract_validate,
    _RawTextIterableDataset,
    _wrap_split_argument,
    _create_dataset_directory,
    _read_text_iterator,
)

# 各个数据集的行数，调整这里来实验不同的数据集大小对模型翻译效果的影响
NUM_LINES = {
    'train': 290000,
    'valid': 20000,
    'test': 1000,
}

DATASET_NAME = "NEWS-Commentary"

@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'valid', 'test'))
def NEWSCOM(root, split, language_pair=('zh', 'en')):
    src_path = os.path.join(root, 'news_zh.txt')
    trg_path = os.path.join(root, 'news_en.txt')

    src_data_iter = _read_text_iterator(src_path)
    trg_data_iter = _read_text_iterator(trg_path)

    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], zip(src_data_iter, trg_data_iter))
