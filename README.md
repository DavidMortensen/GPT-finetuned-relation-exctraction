# GPT-finetuned-relation-exctraction
Improving Relationship Extraction with pre-trained Transformer Language Models (Open AI GPT)

## Installation

Clone repo to local and then download the weights of the OpenAI pre-trained Transformer with the following command: 

```
./download-model.sh
```

## Preperation of the data

The SemEval data is already correctly formated in the repo, however, the tacred dataset can be downloaded from LDC(link) and will need to be formated using: 
```bash
python data_converter.py <DATASET DIR> <CONVERTED DATASET DIR>
```
