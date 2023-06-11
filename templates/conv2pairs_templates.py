import argparse
import os
from glob import glob
import random
import json
import pip
import datetime
import numpy as np
import pandas as pd
from IPython.display import display

# load input files
def loadBiasSpecFile(bias_spec_file):
  with open(bias_spec_file, 'r') as file:
    bias_spec = json.load(file)

  return bias_spec

# extract issue categories per column
def cat2oneHot(row, all_cats):
  cat_list = []
  if row['discarded'] == 1:
    cat_list = [c.strip() for c in str(row['discard_cats']).split(',')]

  n_cat_list = []
  for cat in cat_list:
    cname = ''
    if cat == "10":
      cname = "A"
    else:
      cname = cat
    n_cat_list.append(cname)
  
  return pd.Series({f'C{cat}': 1 if str(cat) in n_cat_list else 0 for cat in all_cats })

def get_words(bias):
  t1 = list(bias['social_groups'].items())[0][1]
  t2 = list(bias['social_groups'].items())[1][1]
  a1 = list(bias['attributes'].items())[0][1]
  a2 = list(bias['attributes'].items())[1][1]

  #(t1, t2, a1, a2) = make_lengths_equal(t1, t2, a1, a2)

  return (t1, t2, a1, a2)

def get_group_term_map(bias):
  grp2term = {}
  for group, terms in bias['social_groups'].items():
    #print(group, "terms:", terms)
    grp2term[group] = terms

  return grp2term

def get_att_term_map(bias):
  att2term = {}
  for att, terms in bias['attributes'].items():
    #print(att, "terms:", terms)
    att2term[att] = terms

  return att2term

def convert2pairsBaseline(bias_spec):
  pairs = []
  headers = ['group','group_term','template', 'att_term_1','att_term_2', 'label_1','label_2','discarded']

  nb = 0
  for bias in bias_spec:
    name = bias['name']
    print(name)
    X, Y, A, B = get_words(bias)

    XY_2_xy = get_group_term_map(bias)
    #print(f"grp2term: {XY_2_xy}")
    AB_2_ab = get_att_term_map(bias)
    #print(f"att2term: {AB_2_ab}")

    #print("Templates:", bias['templates'])
    #print("Group terms:", list(XY_2_xy.items())[0][1])
    #print("Group terms:", list(XY_2_xy.items())[1][1])

    # Stereotype first
    #print(list(AB_2_ab.items())[0][1])
    for att_term in list(AB_2_ab.items())[0][1]:
      for grp1_term, grp2_term in zip(list(XY_2_xy.items())[0][1], list(XY_2_xy.items())[1][1]):
        #print(f"Att: {att_term} -> grp1: {grp1_term}, gpr2: {grp2_term}")
        for template in bias['templates']:
          pairs.append([name, att_term, template.replace("[A]",att_term).replace("[T]","[MASK]"), 
                        grp1_term, grp2_term, "stereotype", "anti-stereotype", 0])
          
    # Anti-stereotype first
    for att_term in list(AB_2_ab.items())[1][1]:
      for grp1_term, grp2_term in zip(list(XY_2_xy.items())[0][1], list(XY_2_xy.items())[1][1]):
        #print(f"Att: {att_term} -> grp1: {grp1_term}, gpr2: {grp2_term}")
        for template in bias['templates']:
          pairs.append([name, att_term, template.replace("[A]",att_term).replace("[T]","[MASK]"), 
                        grp1_term, grp2_term, "anti-stereotype", "stereotype", 0])

    #nb+=1
    #if nb>0:
    #  break

  bPairs_df = pd.DataFrame(pairs, columns=headers)
  bPairs_df = bPairs_df.drop_duplicates(subset = ["group", "group_term","att_term_1","att_term_2","template"])

  return bPairs_df

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some arguments')
  parser.add_argument('--bias_spec_json', type=str, required=True, help="bias specification, needed to")
  parser.add_argument('--out_path', type=str, required=True, help="Outpur direcrtory to put csv sentence pairs into")

  args = parser.parse_args()
  print("Args:", args)

  # Load bias specification to extract all group terms
  bias_specs = loadBiasSpecFile(args.bias_spec_json)

  # create headers for discard reason categories
  all_cats = list(range(10))[1:]
  all_cats.append('A')

  # Convert 2 Pairs
  biasPairs_df = convert2pairsBaseline(bias_specs)
  print("Length:", biasPairs_df.shape)

  # expand reasons for discard into individual columns
  biasPairs_df[all_cats] = biasPairs_df.apply(cat2oneHot, all_cats=all_cats, axis=1)
  save_filename = os.path.join(args.out_path, '_'.join(os.path.basename(args.bias_spec_json).split('_')[0:3])+"_pairs.csv")
  os.makedirs(os.path.dirname(save_filename), exist_ok=True)
  
  biasPairs_df.to_csv(save_filename)




