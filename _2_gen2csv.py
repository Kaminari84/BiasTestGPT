import argparse
import os
import re
import sys
import random
import json
import pip
import datetime
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
tqdm().pandas()

import string

# Adding period to end sentence
def add_period(template):
  if template[-1] not in string.punctuation:
    template += "."
  return template

def sentence_to_template(sentence_json):  
  sentence = sentence_json['sentence']
  grp_term = sentence_json['group_term']
  template = sentence

  fnd_grp = list(re.finditer(f"(^|[ ]+){grp_term.lower()}[ .,!]+", template.lower()))
  while len(fnd_grp) > 0:
    idx1 = fnd_grp[0].span(0)[0]
    if template[idx1] == " ":
      idx1+=1
    idx2 = fnd_grp[0].span(0)[1]-1
    template = template[0:idx1]+f"[T]"+template[idx2:]

    fnd_grp = list(re.finditer(f"(^|[ ]+){grp_term.lower()}[ .,!]+", template.lower()))

  return template

# cleaning weird symbols
def replaceSymbols(sentence, symbols):
  for sym in symbols:
    sentence = sentence.replace(sym,"")
    sentence = sentence.replace(sym.lower(),"")
  return sentence

def export_generations_to_csv(filepath, dest_folder):
  with open(filepath, "r") as file:
    generations = json.load(file)
  
  result = []
  symbol_list = ["~","#","*","?","Â","¬","","«","»","Ã","¢","”","€","Š","· ",""," ",""]
  headers = ['Bias Name', 'Attribute word', 'Group Term', 'Generated sentence', 'Template version', 'Discarded', 'Reason for discard']
  for bias in generations:
      name = bias['name']
      for att_group in list(bias.keys())[1:]:
          for term in list(bias[att_group].keys()):
              for item in bias[att_group][term]['sentences']:
                  sentence = replaceSymbols(item['sentence'], symbol_list)
                  grp_term = item['group_term']
                  att_term = item['attribute_term']
                  template = replaceSymbols(add_period(sentence_to_template(item)), symbol_list)
                  result.append([name, att_term, grp_term, sentence, template, None, None])

  df = pd.DataFrame(result, columns=headers)

  path = os.path.normpath(filepath)
  filename_root = path.split(os.sep)[-1].replace(".json", ".csv")
  df.to_csv(f"{dest_folder}/{filename_root}", encoding='utf-8')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some arguments')
  parser.add_argument('--source_path', type=str, required=True, help="Source directory with generation json files")
  parser.add_argument('--out_path', type=str, required=True, help="Destination directory to put the csv files into")

  args = parser.parse_args()
  print("Args:", args)


  # convert files from folder
  gen_files = glob(os.path.join(args.source_path, "*.json"))
  print(f"Gen files: {len(gen_files)}")
  
  # create output directory if needed
  if len(gen_files)>0:
    os.makedirs(args.out_path, exist_ok=True)

  # process all the files in source directory
  for gen_file in gen_files:
    print("File:", os.path.basename(gen_file))

    export_generations_to_csv(gen_file,
      args.out_path)