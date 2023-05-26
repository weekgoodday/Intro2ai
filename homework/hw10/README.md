# 作业要求
## Transformer神经机器翻译（NMT）
代码结构：
```
├── .data # 数据集
│   ├── Multi30k # 德语-英语数据集
│   ├── News-Commentary # 中文-英语数据集
├── attention.py # Attention模块
├── transformer.py # Transformer模块
├── dataset_zh2en.py # 中文-英语数据集处理
├── nmt_zh2en.py # 中文-英语翻译训练
├── nmt_de2en.py # 德语-英语翻译训练
```
### 作业任务：
1. 阅读`attention.py`和`transformer.py`代码，理解Transformer模型的结构。
2. 阅读`nm_zh2en.py`理解训练过程。
3. 运行`nmt_zh2en.py`，训练中文-英语翻译模型。训练10个epoch，试验不同的源语言语句的翻译效果。更改模型的超参数，观察模型的翻译效果。
4. 修改`dataset_zh2en.py`，更改训练集和测试集的大小，重新训练模型，观察模型的翻译效果。
5. 运行`nmt_de2en.py`，训练德语-英语翻译模型。训练10个epoch，试验不同的源语言语句的翻译效果。

### 作业说明

3、4、5均需在实验报告内描述你的观察结果和翻译截图。

提交：代码（除去data文件和、whl文件），实验报告

## 安装配置
主要依赖参考版本
```
torch               1.9.0
torch-mlu           1.6.0-torch1.9 # 仅在MLU上运行时需要
torchtext           0.10.1
spacy               3.5.2
spacy-legacy        3.0.12
spacy-loggers       1.0.4
spacy-pkuseg        0.0.32
```

torch安装：

```bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
或
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

从本地`.whl`文件安装spacy的分词模型

```
pip install zh_core_web_sm-3.5.0-py3-none-any.whl
pip install en_core_web_sm-3.5.0-py3-none-any.whl
pip install de_core_news_sm-3.5.0-py3-none-any.whl
```

