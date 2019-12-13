

<h1>
<p>Deep Learning of a Topic 
</h1>




| Section | 
|-|
| [Reference Repository](#reference-repository) | 
| [Modifications](#modifications) | 
| [Tasks](#tasks) | 
| [Adding more topic](#more-topic) | 
| [Requirements](#requirements) | 

## Reference repository

[🤗 Transformers](https://github.com/huggingface/transformers) It provided general-purpose architectures and pretrained models for BERT and other models.


## Modifications

HuggingFace provides easy to use scripts that performed the tasks that we wanted to do. The original repository required little modifications as we wanted to research with the data and the architecture and not the tasks. 

We have added the `deep_reading/topics` section contains the topics that we used to research our ideas.

`deep_reading` folder stores all the models that were created in the research phase with the structure as `topic_epochs` and `topic_epochs/squad` containing the 'SQUAD' files.  

[Google Drive: 926GB and counting](https://drive.google.com/drive/folders/1MjR9Xrp867WWiNNBzTObjayZPvPNKfzM?usp=sharing) We tried lot of options! The notebooks are in the `Google Colab` folder

We created the `squad.py` and `lm.py` scripts to perform the respective tasks.  

## Tasks

##### Baseline Question Answering Model
For creating a baseline Question Answering model 

```bash
python squad.py --train
```

This trains the model `bert-base-cased` and stores in `deep_learning/bert-base-cased/squad`

##### Masked Language Modelling on "Imperialism"
```bash
python lm.py --topic Imperialism --epochs 16
```
This will do the LM task on the topic Imperialism for 16 epochs. The other parameter were kept same for comparisons.
This save the model in `deep_reading/bert-base-cased-Imperialism_16`

##### Question Answering Model task for the above model


```bash
python squad.py --topic Imperialism --epochs 16 --train 
```

##### Evaluating the model
To test this model

For running the squad on `Imperialism` topic on the baseline model `bert-base-cased` (trained on SQUADv1 training dataset)
```bash
python squad.py --topic Imperialism --epochs 16 
```

The result for this is 

```bash
  "exact": 38.18681318681319,
  "f1": 42.47042552170011,
  "total": 364,
  "HasAns_exact": 74.33155080213903,
  "HasAns_f1": 82.66970529357668,
  "HasAns_total": 187,
  "NoAns_exact": 0.0,
  "NoAns_f1": 0.0,
  "NoAns_total": 177
```
For running the squad on `Imperialism` topic on our model

```bash
python squad.py --topic Imperialism --epochs 16 --bert_topic
```

This give 
```bash
  "exact": 38.73626373626374,
  "f1": 43.21981526655647,
  "total": 364,
  "HasAns_exact": 75.40106951871658,
  "HasAns_f1": 84.12841046538266,
  "HasAns_total": 187,
  "NoAns_exact": 0.0,
  "NoAns_f1": 0.0,
  "NoAns_total": 177
```

### More topic

To add more topics,


##### Find sentences for LM task "Topic.txt"
1: Extract paragraphs from the internet for the topic.

2: Separate them on different lines. Find: ```([.])( )*``` Replace ```\1\n``` in Notepad++

3: Remove citations Find ```([\[])([0-9])*[\]]``` Replace ``` ```

4: Remove Empty lines.

##### Find questions for LM task "Topic.json"

1. ```less dev-v2.0.json | jq '.data[] | select(.title=="topic")' > Topic.json``` where `dev-v2.0.json` is the squad evaluating dataset.

2. To the `Topic.json` file add `{ 	"data": [ 		{` to the start and `	], 	"version": "2.0" }` to the end.

2. Place the `Topic.json` and `Topic.txt` in the topics folder. 


### Requirements

The above project was run on google colab with

- Python v3.6.9
- GPU 'Tesla K80'
- CUDA 10.0 

The other requirements for this projects is given in the `requirements.txt`


