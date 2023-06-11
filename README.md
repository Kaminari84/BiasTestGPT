# BiasTestGPT
Repository for BiasTestGPT - Using ChatGPT for Social Bias Testing of Language Models (Link TBA)


The following repository contains the BiasTestGPT framework which uses ChatGPT controllable sentence generation for creating dynamic datasets for testing social biases in Pretrained Language Models (PLMs). It includes the 1) test ssentence generation script which can leverage any generators PLM, in particular ChatGPT as long as OpenAI key is provided. It also also contains scripts for all the processing steps from generation to bias testing. These steps, after generation, include 2) conversion of json generations into easily inspectable csv format, 3) turning generated sentences in test sentence templates with controlled social group placeholder, and 4) bias estimates per sentence, per tested attribute and per model using Stereotype Score (% of stereptyped choices in stereotype/anti-stereotype sentence pairs) metric from [Nadeem'20](https://arxiv.org/abs/2004.09456) (stereotype/anti-stereotype pairs).


## BiasTestGPT Generation Framework Steps
Here we describe the steprs followed by the framework. One of the reasons for splitting the generaion into multile steps with separate scripts is to allow for easy inspection of generated sentences as well as for supporting modeular architecture where some steps can be improved with further techniques (e.g., a better approach of turning sentences into templates).

#### Step 1: Generating Test Sentences Using Provided Bias Specifciation
Generation of test sentences requires a bias specification in JSON format on inout, please refer to **./custom_biases/custom_biases_spec.json** for an example of such specification. <OPENAI-TOKEN> needs to be provided by the user.
```
python3 _1_gen_test_sentences.py --bias_spec_json ./custom_biases/custom_biases_spec.json --generator_model 'gpt-3.5-turbo' --out_path './custom_biases/gen_json' --openai_token <OPENAI-TOKEN>
```

#### Step 2: Turn JSON generations into CSV for potential inspection of generated sentences
Starting from generations in JSON format (see example in **/core_biases/gen_json/**), this step generates a CSV version of the generated sentences along with additional columns for potential human annotation - *`Discarded'* and *`Reason for discard'*. The script will process all .json files in a given directory.
```
 python3 _2_gen2csv.py --source_path ./custom_biases/gen_json --out_path ./custom_biases/gen_csv 
```

#### Step 3: Turn CSV templates into stereotype/anti-stereotype pairs
This step turns the csv template sentence output from previous step into stereotype/anti-stereotype pairs. It also preserves any human annotation.
```
python3 _3_csv2pairs_rule.py --source_path ./custom_biases/gen_csv --bias_spec_json ./custom_biases/custom_biases_spec.json --out_path ./custom_biases/gen_pairs_csv
```

#### Step 4: Test Social Bias on given **Tested Model** using Stereotype Score metric from [Nadeem'20](https://arxiv.org/abs/2004.09456)
The tested model accepts paths from HuggingFace Transformer library, examples: *"bert-base-uncased", "bert-large-uncased", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"*
```
python3 _4_ss_test_rule.py --gen_pairs_path ./tmp/pairs --tested_model 'bert-base-uncased' --out_path ./tmp/ss_test
```
