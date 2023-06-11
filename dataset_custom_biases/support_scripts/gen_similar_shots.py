import argparse
import os
import random
import json
import pip
import heapq
from scipy.spatial import distance
import gensim.downloader as api

def install(package):
   pip.main(['install', package])

try:
    import gensim
except ModuleNotFoundError:
    print("The module is not installed")
    install("gensim==3.6.0") # the install function from the question

# load input files
def loadBiasSpecFile(bias_spec_file):
  with open(bias_spec_file, 'r') as file:
    bias_spec = json.load(file)

  return bias_spec

# load stereo-set
def loadStereoSetExamples(stereoset_path):
  with open(stereoset_path, 'r') as file:
    stereoset = json.load(file)

  return stereoset

# Show all available models in gensim-data
def listGensimModels():
  return list(api.info()['models'].keys())

def in_vocab(word, w2v_model):
  try:
    x = w2v_model[word]
    return True
  except KeyError:
    return False
  
def shot_relevance(target, attribute, shot):
    """
    Computes relevance as average cosine similarity between target term with
    the shot group term and attribute term with the shot attr term.
    
    Inputs:
        target: target word desired for generation
        attribute: attribute word desired for generation
        shot: dictionary with keys
            {   Keywords: [],
                Sentence: "",
                group_term: "",
                attribute_term: ""
            }
    Assume at least one of target and attribute is in w2v vocab 
    """
    total_distance = 0
    average = True
    target = target.lower().replace(' ', '').replace('-', '')
    attribute = attribute.lower().replace(' ', '').replace('-', '')
    try:
        # targ_distance =  distance.cosine(w2v_model[target], w2v_model[shot['group_term'].lower().replace(' ', '').replace('-', '')])
        targ_distance =  distance.cosine(w2v_model[target], w2v_model[shot['Keywords'][0].lower().replace(' ', '').replace('-', '')])
        total_distance += targ_distance
    except KeyError:
        targ_distance = None
        average = False
    try:
        # attr_distance = distance.cosine(w2v_model[attribute], w2v_model[shot['attribute_term'].lower().replace(' ', '').replace('-', '')])
        attr_distance = distance.cosine(w2v_model[attribute], w2v_model[shot['Keywords'][1].lower().replace(' ', '').replace('-', '')])
        total_distance += attr_distance
    except KeyError:
        attr_distance = None
        if not average:
            return None, None, None
        else:
            average = False
    
    avg_distance = total_distance
    if average:
        avg_distance /= 2
    
    return avg_distance, targ_distance, attr_distance

def top_similar_shots(target, attribute, shots, n, bias_name):
    """
    Inputs:
        target: string, target word
        attribute: string, attribute word
        shots: list of dictionaries with keys
            {   "Keywords": [kwd1, kwd2],
                "Sentence": sentence using kwd1 and kwd2,
                "group_term": kwd1,
                "attribute_term": kwd2
            }
        n: number of most similar shots
    Return:
        the n most relevant sentences
    """
    if not (in_vocab(target, w2v_model) or in_vocab(attribute, w2v_model)):
    # choose n random shots
        # with open('stereoset-similar-shots-per-bias.json', 'r') as file:
        #     similar_shots = json.load(file)
        # for bias in similar_shots:
        #     if bias['name'] == bias_name:
        #         shots = bias['sentences']
        #         break
        choices = random.sample(shots, n)
        for shot in choices:
            shot = shot.copy()
            shot['avg_cos_distance'] = None
            shot['targ_cos_distance'] = (None, target)
            shot['attr_cos_distance'] = (None, attribute)
        return choices
    relevances = []
    ecount = 0
    for shot in shots:
        avg_distance, targ_distance, attr_distance = shot_relevance(target, attribute, shot)
        if not avg_distance:
            continue
        shot = shot.copy()
        shot['avg_cos_distance'] = avg_distance
        shot['targ_cos_distance'] = (targ_distance, target)
        shot['attr_cos_distance'] = (attr_distance, attribute)
        relevances.append((avg_distance, ecount, shot))
        ecount += 1
    return [x[2] for x in heapq.nsmallest(n, relevances)]

def make_lengths_equal(t1, t2, a1, a2):
  if len(t1) > len(t2):
    t1 = random.sample(t1, len(t2))
  elif len(t1) < len(t2):
    t2 = random.sample(t2, len(t1))

  if len(a1) > len(a2):
    a1 = random.sample(a1, len(a2))
  elif len(a1) < len(a2):
    a2 = random.sample(a2, len(a1))

  return (t1, t2, a1, a2)

def get_words(bias):
  t1 = list(bias['social_groups'].items())[0][1]
  t2 = list(bias['social_groups'].items())[1][1]
  a1 = list(bias['attributes'].items())[0][1]
  a2 = list(bias['attributes'].items())[1][1]

  (t1, t2, a1, a2) = make_lengths_equal(t1, t2, a1, a2)
  
  return (t1, t2, a1, a2)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some arguments')
  parser.add_argument('--bias_spec_json', type=str, required=True, help="Bias specification, needed to")
  parser.add_argument('--example_sentences_json', type=str, required=True, help="Example Test sentences to pick from")
  parser.add_argument('--out_file_json', type=str, required=True, help="Destination directory to put the csv files into")

  args = parser.parse_args()
  print("Args:", args)

  # Load bias specification to extract all group terms
  bias_specs = loadBiasSpecFile(args.bias_spec_json)
  
  # Load stereoset sentences
  example_sentences = loadStereoSetExamples(args.example_sentences_json)
  print(f"Example sentences len: {len(example_sentences)}")

  # Check available gensim models
  gensim_models = listGensimModels()
  print(f"Gensim models: {gensim_models}")

  model_name = 'glove-wiki-gigaword-300'
  w2v_model = api.load(model_name)  

  res = {}
  # Go through biases and get most similar shots
  for bias in bias_specs:
    print(f"Bias-name: {bias['name']}")

    (t1, t2, a1, a2) = get_words(bias)
    
    targets = t1 + t2
    attributes = a1 + a2

    for attr in attributes:
      for targ in targets:
        if attr not in res:
          res[attr] = {}
        res[attr][targ] = top_similar_shots(targ, attr, example_sentences, 10, bias['name'])
    
    with open(args.out_file_json, "w") as file:
       json.dump(res, file, indent=4)



