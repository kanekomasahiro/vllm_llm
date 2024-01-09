# Simple LLMs with vLLM

## Introduction

This project provides a straightforward implementation of Large Language Models (LLM) using the [vLLM](https://github.com/vllm-project/vllm) library.

## Installation

- Python Version: The code is compatible with Python version `3.9.5`.
- Dependencies: Install all the necessary dependencies using the following command:
```sehll
pip install -r requirements.txt
```

## Usage

To download the sample data, consisting of 100 sentences from the [PTB dataset](https://huggingface.co/datasets/ptb_text_only), use the following command.

```shell
curl -X GET \
     "https://datasets-server.huggingface.co/rows?dataset=ptb_text_only&config=penn_treebank&split=train&offset=0&length=100" > data/ptb.json

```

To run the model, use the following command. Ensure to replace `path/to/your/data` and `path/to/your/output/dir` with your actual data and output directories.

```shell
export CUDA_VISIBLE_DEVICES=0
python src/llm.py --data_path path/to/your/data --output_file path/to/your/output/file.jsonl
```

## License

Refer to the LICENSE file in this repository for licensing information.