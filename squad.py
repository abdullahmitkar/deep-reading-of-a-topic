
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertModel, BertTokenizer
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
import os
import argparse


def run_squad(topic,epoch, base=True,train=False):
  predict=topic + ".json"
  if not base:
    model_name="deep_reading/bert-base-cased-" + topic + "_" + str(epoch)
    out_folder="deep_reading/bert-base-cased-" + topic + "_" + str(epoch) + "/squad" 
  else:
    model_name="deep_reading/bert-base-cased"
    out_folder="deep_reading/bert-base-cased/squad"

  
  
  if train:
    print("Training for Question Answering using topic as ", topic, " base folder", model_name, " output directory as", out_folder)
    
    command = "python examples/run_squad.py --log_info 0 \
    --model_type bert \
    --model_name_or_path "+model_name+" \
    --do_train \
    --do_eval \
    --train_file deep_reading/topics/train-v1.1.json \
    --predict_file deep_reading/topics/"+predict +"\
    --overwrite_output_dir\
    --per_gpu_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir "+out_folder
    os.system(command)
  else:
    print("Evaluating for Question Answering using topic as ", topic, " base folder", model_name, " output directory as", out_folder)
    
    command = "python examples/run_squad.py --log_info 0 \
    --model_type bert \
    --model_name_or_path "+model_name+" \
    --do_eval \
    --do_lower_case \
    --train_file deep_reading/train-v1.1.json \
    --predict_file deep_reading/topics/"+predict+" \
    --per_gpu_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir "+out_folder
    os.system(command)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--topic", default=None, type=str, required=True,
                        help="Topic to train squad on")
    parser.add_argument("--epochs", default=None, type=str, required=True,
                        help="No of epochs this LM task was trained for (used for finding the model with name as 'topic_epochs'")
    parser.add_argument("--bert_topic", action='store_true',
                        help="Boolean for run this task on bert-topic - the modified model")
    parser.add_argument("--train", action='store_true',
                        help="Boolean for run this task on bert-base")

    ## Other parameters
    
    args = parser.parse_args()

    

    # Setup distant debugging if needed
    if args.bert_topic:
        base=False
    else:
        base=True
    
    if args.train:
        train=True
    else:
        train=False
    
    run_squad(topic=args.topic, epoch=args.epochs, base=base, train=train)


if __name__ == "__main__":
    main()
