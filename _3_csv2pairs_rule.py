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

# check if term within term list
def checkinList(term, term_list, verbose=False):
  for cterm in term_list:
    if verbose==True: print(f"Comparing <{cterm}><{term}>")
    if cterm == term or cterm.replace(" ","-") == term.replace(' ','-'):
      return True
  return False

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

def _adjustGrpIndex(idx, opp_grp):
  len_opp_group = len(opp_grp)
  if idx >= len_opp_group:
    print(f"Out of bound: {idx}, max: {len_opp_group}")
    idx = random.randint(0,len_opp_group-1)
    print(f"New idx: {idx}")

  return idx

# convert the input csv file to Stereo-Set pairs
def convert2pairs(filename, bias_spec):
  gen_df = pd.read_csv(filename, index_col=0)

  pairs = []
  headers = ['group','group_term','template', 'att_term_1','att_term_2', 'label_1','label_2','discarded','discard_cats']

  for nb, bias in enumerate(bias_spec):
    name = bias['name']
    print(f"Processing bias: {name}")
    X, Y, A, B = get_words(bias)

    XY_2_xy = get_group_term_map(bias)
    #print(f"grp2term: {XY_2_xy}")
    AB_2_ab = get_att_term_map(bias)
    #print(f"att2term: {AB_2_ab}")

    #Bias Name	Attribute word	Group Term	Generated sentence	Template version	Discarded	Reason for discard
    ri = 0
    for idx, row in gen_df[gen_df['Bias Name'] == name].iterrows():
      direction = []
      if checkinList(row['Attribute word'], list(AB_2_ab.items())[0][1]):
        direction = ["stereotype", "anti-stereotype"]
      elif checkinList(row['Attribute word'], list(AB_2_ab.items())[1][1]):
        direction = ["anti-stereotype", "stereotype"]

      grp_term_idx = -1
      grp_term_pair = []
      if row['Group Term'] in list(XY_2_xy.items())[0][1]:
        grp_term_idx = list(XY_2_xy.items())[0][1].index(row['Group Term'])
        grp_term_idx = _adjustGrpIndex(grp_term_idx, list(XY_2_xy.items())[1][1])
        grp_term_pair = [row['Group Term'], list(XY_2_xy.items())[1][1][grp_term_idx]]
      elif row['Group Term'].replace("-"," ") in list(XY_2_xy.items())[0][1]:   
        grp_term_idx = list(XY_2_xy.items())[0][1].index(row['Group Term'].replace("-"," "))
        grp_term_idx = _adjustGrpIndex(grp_term_idx, list(XY_2_xy.items())[1][1])
        grp_term_pair = [row['Group Term'].replace("-"," "), list(XY_2_xy.items())[1][1][grp_term_idx]]
      
      elif row['Group Term'] in list(XY_2_xy.items())[1][1]:
        grp_term_idx = list(XY_2_xy.items())[1][1].index(row['Group Term'])
        grp_term_idx = _adjustGrpIndex(grp_term_idx, list(XY_2_xy.items())[0][1])
        grp_term_pair = [row['Group Term'], list(XY_2_xy.items())[0][1][grp_term_idx]]
        direction.reverse()
      elif row['Group Term'].replace("-"," ") in list(XY_2_xy.items())[1][1]:   
        grp_term_idx = list(XY_2_xy.items())[1][1].index(row['Group Term'].replace("-"," "))
        grp_term_idx = _adjustGrpIndex(grp_term_idx, list(XY_2_xy.items())[0][1])
        grp_term_pair = [row['Group Term'].replace("-"," "), list(XY_2_xy.items())[0][1][grp_term_idx]]
        direction.reverse()
      else:
        print(f"ERROR: Group term {row['Group Term'].replace('-',' ')} does not belong...")
        checkinList(row['Attribute word'], list(AB_2_ab.items())[0][1], verbose=True)
        checkinList(row['Attribute word'], list(AB_2_ab.items())[1][1], verbose=True)

      pairs.append([name, row['Attribute word'], row['Template version'], grp_term_pair[0], grp_term_pair[1], direction[0], direction[1], row['Discarded'], row['Reason for discard']])

  bPairs_df = pd.DataFrame(pairs, columns=headers)
  bPairs_df = bPairs_df.drop_duplicates(subset = ["group", "group_term", "template"])
  display(bPairs_df.head(1))

  return bPairs_df

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some arguments')
  parser.add_argument('--source_path', type=str, required=True, help="Source path with generations in CSV format (use gen2csv.py first)")
  parser.add_argument('--bias_spec_json', type=str, required=True, help="Bias specification, needed to")
  parser.add_argument('--out_path', type=str, required=True, help="Outpur direcrtory to put csv sentence pairs into")

  args = parser.parse_args()
  print("Args:", args)

  # Load bias specification to extract all group terms
  bias_specs = loadBiasSpecFile(args.bias_spec_json)

  # create headers for discard reason categories
  all_cats = list(range(10))[1:]
  all_cats.append('A')

  # process directory
  print(f"Processing directory: {args.source_path}")

  # convert files from folder
  gen_files = glob(os.path.join(args.source_path, "*.csv"))
  print(f"Gen files: {len(gen_files)}")
  if len(gen_files)>0:
    os.makedirs(args.out_path, exist_ok=True)
  for gen_file in gen_files:
    print("File:", os.path.basename(gen_file))
    biasPairs_df = convert2pairs(gen_file, bias_specs)

    # expand reasons for discard into individual columns
    biasPairs_df[all_cats] = biasPairs_df.apply(cat2oneHot, all_cats=all_cats, axis=1)
    biasPairs_df.to_csv(os.path.join(args.out_path, os.path.basename(gen_file).replace(".csv","_pairs.csv")))