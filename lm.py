
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertModel, BertTokenizer
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
import os
import argparse


def run_lm(topic, epoch=4):
  train_file = "deep_reading/topics/"+topic+".txt"
  name=topic +"_"+ str(epoch)
  command="python examples/run_lm_finetuning.py \
  --num_train_epochs="+epoch+" --output_dir=deep_reading/bert-base-cased-"+name+" \
  --overwrite_output_dir --model_type=bert  \
  --model_name_or_path=bert-base-cased \
  --do_train   --train_data_file="+train_file+" --mlm"

  os.system(command)
  


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--topic", default=None, type=str, required=True,
                        help="Topic to train LM task on on")
    parser.add_argument("--epochs", default=None, type=str, required=True,
                        help="No of epochs to run this lm task on")
    
    args = parser.parse_args()

    
    run_lm(topic=args.topic, epoch=args.epochs)


if __name__ == "__main__":
    main()
