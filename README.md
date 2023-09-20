---
language:
 - en
library_name: nemo
tags:
- text generation
- pytorch
- causal-lm
license: llama2

---
# SteerLM Llama-2 13B

<style>
img {
 display: inline;
}
</style>

|[![Model architecture](https://img.shields.io/badge/Model%20Arch-Transformer%20Decoder-green)](#model-architecture)|[![Model size](https://img.shields.io/badge/Params-13B-green)](#model-architecture)|[![Language](https://img.shields.io/badge/Language-Multilingual-green)](#datasets)


## Model Description

SteerLM Llama-2 is a 13 billion parameter generative language model based on the open-source Llama-2 architecture. It has been customized using the SteerLM method developed by NVIDIA to allow for user control of model outputs during inference.

Key capabilities enabled by SteerLM:

- Dynamic steering of responses by specifying desired attributes like quality, helpfulness, and toxicity
- Simplified training compared to RLHF techniques like fine-tuning and bootstrapping

## Model Architecture and Training
The SteerLM method involves the following key steps:

1. Train an attribute prediction model on human annotated data to evaluate response quality
2. Use this model to annotate diverse datasets and enrich training data
3. Perform conditioned fine-tuning to align responses with specified combinations of attributes
4. (Optionally) Bootstrap training through model sampling and further fine-tuning

SteerLM Llama-2 applies this technique on top of the Llama-2 architecture. It was pretrained on internet-scale data and then customized using [OASST](https://huggingface.co/datasets/OpenAssistant/oasst1) and [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) data.
 

## Getting started

To use SteerLM Llama-2, follow these steps:

1. You will need to install NVIDIA Apex and [NeMo](https://github.com/NVIDIA/NeMo). 

```
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 03c9d80ed54c0eaa5b581bf42ceca3162f085327
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./
```

```
pip install nemo_toolkit['nlp']==1.17.0
``` 

Alternatively, you can use NVIDIA NeMo Framework training docker container with all dependencies pre-installed.

2. Launch eval server 

```
git clone https://github.com/NVIDIA/NeMo.git 
cd NeMo/examples/nlp/language_modeling
git checkout v1.17.0
python megatron_gpt_eval.py gpt_model_file=LLAMA2-13B-SteerLM.nemo trainer.precision=16 server=True tensor_model_parallel_size=4 trainer.devices=1 pipeline_model_parallel_split_rank=0
```

3. Send prompts to your model!

```python
import json
import requests

def get_answer(question, max_tokens, values, eval_port='1427'):
    prompt = f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{question}
<extra_id_1>Assistant
<extra_id_2>{values}
"""

    prompts = [prompt]
    data = {
        "sentences": prompts,
        "tokens_to_generate": max_tokens,
        "top_k": 1,
        'greedy': True,
        'end_strings': ["<extra_id_1>", "quality:", "quality:9", "quality:0"]
    }

    url = f"http://localhost:{eval_port}/generate"
    response = requests.put(url, json=data)
    json_response = response.json()

    response_sentence = json_response['sentences'][0][len(prompt):]
    return response_sentence


def encode_labels(labels):
    items = []
    for key in labels:
        value = labels[key]
        items.append(f'{key}:{value}')
    return ','.join(items)

values = OrderedDict([
    ('quality', 9),
    ('toxicity', 0),
    ('humor', 0),
    ('creativity', 0),
    ('violence', 0),
    ('helpfulness', 9),
    ('not_appropriate', 0),
])
values = encode_labels(values)

question = """Where and when did techno music originate?"""

print(get_answer(question, 4096, values))
```


## Evaluation results

[MT-bench](https://arxiv.org/abs/2306.05685) evaluation results:

|Category | score|
|---|---|
|total|  6.13|
|writing | 7.8|
|roleplay | 8.15|
|extraction | 5.52|
|stem | 8.43|
|humanities | 9.02|
|reasoning | 4.95|
|math | 2.15|
|coding | 3.0|

## Limitations

Meta’s Llama2 model was trained on publicly available data sources that could include unsafe content. See Meta's Llama2 paper, section 4.1, "Safety in Pretraining" for more details: https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/. 
The model may amplify unsafe content, especially when prompted with unsafe content. NVIDIA did not perform bias or toxicity removal or model alignment on the Llama2 model. NVIDIA’s SteerLM methodology applied to Llama2 provides the opportunity to improve model quality through a fine-tuning technique based on data annotation of specific important categories and allows adjustments to model output at run-time based on those same categories.




## License

- Llama 2 is licensed under the [LLAMA 2 Community License](https://ai.meta.com/llama/license/), Copyright © Meta Platforms, Inc. All Rights Reserved.
- Your use of the Llama Materials must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the [Acceptable Use Policy](https://ai.meta.com/llama/use-policy) for the Llama Materials.