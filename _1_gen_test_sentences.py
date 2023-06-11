import argparse
import os
import json
import random
import torch
import string
import time
import backoff  # for exponential backoff

from tqdm import tqdm
tqdm().pandas()

from transformers import AutoModelForCausalLM, AutoTokenizer
from genChatGPT import initOpenAI, genChatGPT

# fixed shots optionally used in case no dynamic shots are available
fixed_shots = [
    {"Keywords": ["dog","frisbee", "catch", "throw"], "Sentence": "A dog leaps to catch a thrown frisbee"},
    {"Keywords": ["apple", "bag", "puts"], "Sentence": "A girl puts an apple in her bag"},
    {"Keywords": ["apple", "tree", "pick"], "Sentence": "A man picks some apples from a tree"},
    {"Keywords": ["apple", "basket", "wash"], "Sentence": "A boy takes an apple from a basket and washes it"}
]

# Adding period to end sentence
def add_period(template):
  if template[-1] not in string.punctuation:
    template += "."
  return template

def getModel(model_name, device):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name) #other variations - gpt-neo-125M, gpt-neo-1.3B, gpt-neo-2.7B

  model = model.to(device)
  model.eval()
  torch.set_grad_enabled(False)

  return model, tokenizer

# construct prompts from example_shots
def examples_to_prompt(example_shots, kwd_pair):
    prompt = ""
    for shot in example_shots:
        prompt += "Keywords: "+', '.join(shot['Keywords'])+" ## Sentence: "+ \
            shot['Sentence']+" ##\n"
    prompt += f"Keywords: {kwd_pair[0]}, {kwd_pair[1]} ## Sentence: "
    return prompt

# few-shot prompted generations
def genTransformer(kwd_pair, count, example_shots, temperature=0.8):
    # construct prompts
    prompt = examples_to_prompt(example_shots, kwd_pair)
    
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(device)

    #sample only from 80% most likely words
    sample_output = model.generate(
                              input_ids,
                              pad_token_id=tokenizer.eos_token_id,
                              do_sample = True,
                              max_length = input_ids[0].shape[0]+30,
                              temperature = temperature,
                              top_k = 50,
                              top_p = 0.85,
                              num_return_sequences = count,
                              early_stopping = True
    )

    resp = []
    for gen in sample_output:
        text = tokenizer.decode(gen[input_ids[0].shape[0]:], skip_special_tokens = True)

        # decide on the end of the generation
        min_end = min(text.find("##"), text.find("Keywords:"))

        # filter out generation that contain no keywords, remove weird characters
        if min_end != -1:
            sentence = text[0:min_end].strip(' \xa0A\xa0a\t').replace("_", "").replace("*", "").replace("~", "")
            split = sentence.lower().replace('.', '').split()
            if kwd_pair[0].lower() in split and kwd_pair[1].lower() in split:
                resp.append({'sentence': sentence, 'group_term': kwd_pair[0], 'attribute_term': kwd_pair[1]})

    return resp

# load bias specification file
def loadBiasSpecFile(bias_spec_file):
  with open(bias_spec_file, 'r') as file:
    bias_spec = json.load(file)

  return bias_spec

# load precalculated similar shots
def loadSimilarShotsFile(similar_shots_file):
  with open(similar_shots_file, "r") as file:
    similar_shots = json.load(file)

  return similar_shots

# make sure to use equal number of keywords for opposing attribute and social group specifications
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

# get bias specification keywords
def get_words(bias):
  t1 = list(bias['social_groups'].items())[0][1]
  t2 = list(bias['social_groups'].items())[1][1]
  a1 = list(bias['attributes'].items())[0][1]
  a2 = list(bias['attributes'].items())[1][1]

  (t1, t2, a1, a2) = make_lengths_equal(t1, t2, a1, a2)

  return (t1, t2, a1, a2)

def top_similar_shots(target, attribute, similar_shots, n):
  global fixed_shots

  #attribute = attribute.replace('-', ' ')
  assert n <= 10
  #print(f"Attribute: {attribute}, Target: {target}")
  try:
    if attribute in similar_shots:
      if target in similar_shots[attribute]:
        return similar_shots[attribute][target][0:n]
      else:
        return similar_shots[attribute][target.replace('-',' ')][0:n]
    else:
      attribute = attribute.replace('-', ' ')
      if target in similar_shots[attribute]:
        return similar_shots[attribute][target][0:n]
      else:
        return similar_shots[attribute][target.replace('-',' ')][0:n]
  except KeyError as err:
    print(f"Failed retrieving [{attribute}][{target}], using fixed shots: {err}")
    
  return fixed_shots

# generate sentences for single attribute keyword and 2 social group keywords
def single_attr_generations(t1, t2, att_term, shot_technique, shot_repository, gen_fn=None, num_shots=5, output_batch_size=5, temperature=0.8):
  att_term = att_term#.replace(' ', '-')
  att_gens = []
  t1_terms_tried = []
  t2_terms_tried = []
  t1_num_sentences_generated = 0
  t2_num_sentences_generated = 0
  MAX_TRIES = 40

  # target group 1
  tmp_gens = []
  tries = 0
  while len(tmp_gens) < 1 and tries < MAX_TRIES:
    grp_term = random.choice(t1)
    grp_term = grp_term#.replace(' ', '-')
    t1_terms_tried.append(grp_term)

    shots_1 = shot_repository
    if shot_technique == "similarity":
      shots_1 = top_similar_shots(grp_term, att_term, shot_repository, num_shots)
      
    tmp_gens = gen_fn([grp_term, att_term], output_batch_size, shots_1, temperature)
    tries += 1
    print(".", end='', flush=True)
    # print(f"At {gen_num} of {total_to_gen}, Generations for {[grp_term, att_term]}: {tmp_gens}")
  att_gens.extend(tmp_gens)
  t1_num_sentences_generated += tries * output_batch_size

  # target group 2
  tmp_gens = []
  tries = 0
  while len(tmp_gens) < 1 and tries < MAX_TRIES:
    grp_term = random.choice(t2)
    grp_term = grp_term#.replace(' ', '-')
    t2_terms_tried.append(grp_term)
    
    shots_2 = shot_repository
    if shot_technique == "similarity":
      shots_2 = top_similar_shots(grp_term, att_term, shot_repository, num_shots)
    
    tmp_gens = gen_fn([grp_term, att_term], output_batch_size, shots_2, temperature)
    tries += 1
    print(".", end='', flush=True)
  att_gens.extend(tmp_gens)
  t2_num_sentences_generated += tries * output_batch_size
  
  return {"sentences": att_gens,
          "t1_shots": shots_1,
          "t1_terms_tried": t1_terms_tried,
          "t1_num_sentences_generated": t1_num_sentences_generated,
          "t2_shots": shots_2,
          "t2_terms_tried": t2_terms_tried,
          "t2_num_sentences_generated": t2_num_sentences_generated
          }

def genTestSentences(bias_spec, shot_technique, shot_repository, gen_fn, save_path):
  generations = []

  dir_name = os.path.dirname(save_path)
  file_name = f"_temp_{os.path.basename(save_path)}"
  temp_path = os.path.join(dir_name, file_name)
  print(f"Checking to temp path: {temp_path}")

  os.makedirs(os.path.dirname(temp_path), exist_ok=True)

  if os.path.exists(temp_path):
    print("Save exists, loading...")
    # load the generated test sentences
    with open(temp_path, 'r') as file:
      generations = json.load(file)
      print(f"Loaded {len(generations)} biases")

  for bn, bias in enumerate(bias_spec):
    print(f"[{bn}] {bias['name']}")
    skip = False
    for sb in generations:
      if bias['name'] == sb['name']:
        print(f"  Skipping bias <{sb['name']}>...")
        skip = True
        break

    if skip == True:
      continue

    (t1, t2, a1, a2) = get_words(bias)
    a1_name = list(bias['attributes'].items())[0][0]
    a2_name = list(bias['attributes'].items())[1][0]
    a1_generations = {}
    a2_generations = {}

    for an, att_term in enumerate(a1):
        a1_generations[att_term] = single_attr_generations(t1, t2, att_term, shot_technique, shot_repository,
                                                           gen_fn = gen_fn,
                                                           num_shots=5, output_batch_size=5,
                                                           temperature=0.8)
        print(f"\nAttr [{an} of {len(a1)}]:<{att_term}> -> {len(a1_generations)}")
    #print(a1_generations)
    
    for an, att_term in enumerate(a2):
        a2_generations[att_term] = single_attr_generations(t1, t2, att_term, shot_technique, shot_repository,
                                                           gen_fn = gen_fn,
                                                           num_shots=5, output_batch_size=5,
                                                           temperature=0.8)
        print(f"\nAttr [{an} of {len(a2)}]:<{att_term}> -> {len(a2_generations)}")
    
    #print(a2_generations)
    
    generations.append({
        'name': bias['name'],
        a1_name: a1_generations,
        a2_name: a2_generations
    })

    print(f"Bias {bias['name']} - Num biases covered: {len(generations)}")
    print(f"Saving to temp path: {temp_path}")
  
    # save the generated test sentences
    with open(temp_path, 'w') as file:
      json.dump(generations, file, indent=4)

  return generations

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some arguments')
  parser.add_argument('--top_shots_json', type=str, required=False, help="Precalculated top most similar shots for bias specification terms, used in few-shot strategy")
  parser.add_argument('--bias_spec_json', type=str, required=True, help="bias specification")
  parser.add_argument('--generator_model', type=str, required=True, help="Name of the text generation model - 'gpt-3.5-turbo' or a HuggingFace path, e.g., EleutherAI/gpt-neo-125M")
  parser.add_argument('--out_path', type=str, required=True, help="Output directory to save json test sentence templates into")
  parser.add_argument('--openai_token', type=str, required=False, help="OpenAI access token, needed for OpenAI generators only")

  args = parser.parse_args()
  print("Args:", args)

  use_openai = False
  if args.generator_model == 'gpt-3.5-turbo' or args.generator_model == "gpt-4":
    use_openai = True
    initOpenAI(key=args.openai_token, 
        mod_name = args.generator_model, 
        prmpt_fn = examples_to_prompt,
        # this instruction is only used with few-shot generation
        inst="Write sentence connecting the given keywords. Use examples as guide for the type of sentences to write.")
  else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = getModel(args.generator_model, device)

  # load bias speification
  bias_specs = loadBiasSpecFile(args.bias_spec_json)
  print(f"Num biases: {len(bias_specs)}")

  # load precalulated shots shots
  shots = fixed_shots
  shot_technique = "fixed"
  if args.top_shots_json != None:
    print("Using precalulated most similar shots...")
    shot_technique = "similarity"
    shots = loadSimilarShotsFile(args.top_shots_json)
    print(f"Num precalculated group terms: {len(list(shots.keys()))}")
  else:
    print("Using fixed shots...")
    shots = fixed_shots

  gen_batch_size = 5
  temperature = 0.8
  num_shots = 5
  if use_openai:
    print(f"Test with OpenAI generator: {args.generator_model}")
    generations = genChatGPT(['man', 'math'], gen_batch_size, fixed_shots, temperature=temperature)
  else:
    generations = genTransformer(['man','math'], gen_batch_size, fixed_shots, temperature=temperature)

  print(generations)

  # change filename before running
  mod2abbr = {'EleutherAI/gpt-j-6b': "gpt-j-6b",
    'EleutherAI/gpt-neo-2.7B': "neo-2.7b",
    'EleutherAI/gpt-neo-1.3B': "neo-1.3b",
    'EleutherAI/gpt-neo-125M': "neo-125m",
    'gpt-3.5-turbo': "gpt-3.5",
    'gpt-4': "gpt-4"}
  
  mod_abbr = mod2abbr[args.generator_model] if args.generator_model in mod2abbr else args.generator_model[-5:0]
  
  # save filepath
  filename = f"{shot_technique}-{mod_abbr}-temp-{temperature}-shots-{num_shots}.json"
  save_path = os.path.join(args.out_path, f'{filename}')

  # generate test sentences
  generations = []
  if use_openai:
    print(f"Generating with OpenAI generator: {args.generator_model}")
    generations = genTestSentences(bias_specs, shot_technique, shots, gen_fn=genChatGPT, save_path=save_path)
  else:
    generations = genTestSentences(bias_specs, shot_technique, shots, gen_fn=genTransformer, save_path=save_path)
  
  print(f"Num generations: {len(generations)}")
  
  # save the generated test sentences
  with open(save_path, 'w') as file:
    json.dump(generations, file, indent=4)

