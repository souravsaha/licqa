'''
Rank the entities based on the sentence similary

Steps

1. Load sentence embedding first >> DONE
2. Read the settings file and process it  >> DONE  
3. Do embedding of question q >> TRIVIAL
4. Do embedding of sentences containing the entities >> WIP
5. Do coreference resolution 
6. Save the file  >> TRIVIAL

'''

'''
Step 1: Load sentence embeddings
'''
# import basic libraries


from InferSent.models import InferSent
from random import randint
import numpy as np
import torch
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
import json
import unicodedata

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# Load model
model_version = 1
MODEL_PATH = "InferSent/encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'InferSent/GloVe/glove.840B.300d.txt' if model_version == 1 else 'InferSent/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)

# Load some sentences TODO : load our sentence
sentences = []

with open('InferSent/samples.txt') as f:
    for line in f:
        sentences.append(line.strip())
print(len(sentences))
"""
with open('../Data/Corpora/CQ_T/top10/CQ-T150-q001.json') as f:
   data = json.load(f)

doc_content = []
for i in range(0, 10):
    # doc_name = data[1]['fname']
    doc_content.append(data[i]['text'])
    doc_content[i] = doc_content[i].replace("\n", " ")
    # doubt whether to remove this or not
    doc_content[i] = remove_accented_chars(doc_content[i])
    doc_content[i] = doc_content[i].lower()
    #sentences.append(sent_tokenize(doc_content[i]))
    #print(type(sent_tokenize(doc_content[i])))
    sentences.extend(sent_tokenize(doc_content[i]))

    #print(len(sentences))
    
    #system.exit(1)
print((sentences))
"""
#### Encode Sentences ########

# gpu mode : >> 1000 sentences/s
# cpu mode : ~100 sentences/s

embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))

np.linalg.norm(model.encode(['the cat eats.']))


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


print(cosine(model.encode(['the cat eats.'])[0], model.encode(['the cat drinks.'])[0]))

# let's get the sentences from Entity and tag 
def get_sentence_from_entity(doc_content, input_ent, tag):
    doc = nlp(doc_content)
    #tags = [(X.text, X.label_, X) for X in doc.ents]
    sentences = []
    for sent in doc.sents:
      text = nlp(sent.text)
      for ent in text.ents:
        if str.lower(ent.text) == str.lower(input_ent) and ent.label_ == tag : 
            sentences.append(sent)

    #position = [(X.start, X.end) for X in doc.ents]
    return sentences

"""
Read the entities and settings file

"""
import yaml
config_file_name = 'configure.yml'

# defined it here too
with open(config_file_name) as config_file:
    config_file_values = yaml.load(config_file)

# Get Entitites for each question

qid = config_file_values["qid"]
quesType = config_file_values["quesType"]
quesPathStart = config_file_values["quesPathStart"]
corpusPathStart = config_file_values["copusPathStart"]
resultPathStart = config_file_values["resultPathStart"]
samplingType = config_file_values["samplingType"]

from extract_entities import get_named_entities, get_named_entities_with_sentence
from pprint import pprint
from common import get_corpus_path, get_question_and_gold_answer
import pickle
import json

import en_core_web_sm
nlp = en_core_web_sm.load()
#import en_core_web_lg
#nlp = en_core_web_lg.load()

nlp.max_length = 1030000

# Get Question type from dump file
map_qid_qtype_pair = {}
#input_file_name = "results/quesType_finetune_" + quesType 
input_file_name = "results/ques_type_use_" + quesType 
with open(input_file_name, 'r') as input_file:
    lines = input_file.readlines() 

for line in lines:
    line = line.replace('\n', '')           # only valid for google question type 
    question_id = line.split("\t")[0]
    #spacy_tag_ques_type = line.split("\t")[2]  hand written rule 3rd column is the spacy
    spacy_tag_ques_type = line.split("\t")[5]   # TODO : change this
    map_qid_qtype_pair[question_id] =  spacy_tag_ques_type

#print(map_qid_qtype_pair)
#exit(1)

# result file map for qid and top ranked entities
result_file_qid_answers = {}
ngpu = 2
for i in range(150):
    i = 121
    ques_id = int(qid) + i
    question, answer, ques_exact_id = get_question_and_gold_answer( ques_id, quesType, quesPathStart)
    
    # hack fotr workaround of query 121 : for T
    #if i == 121 or i == 108:
    #    continue
    #if i == 121:
    #    continue
    # hack:  ques_id looks like cqw-150-q001, corpus id 001, extract from this
    corpus_ques_id = ques_exact_id.split('-')[2][1:]
    print(corpus_ques_id)

    corpus_path = get_corpus_path(corpus_ques_id, quesType, corpusPathStart, samplingType)
    #tags = get_named_entities(ques_id, quesType, quesPathStart, corpus_path)
    tags, ent_sentence_map = get_named_entities_with_sentence(ques_id, quesType, quesPathStart, corpus_path)

    #pprint(tags)
    
    # convert the tags into list (ordered way)
    tags_list = sorted(tags.items(), key=lambda pair: pair[1], reverse=True)
    """
    output_file_name = "results/all_named_entities_" + quesType
    question, answer, ques_exact_id = get_question_and_gold_answer( ques_id, quesType, quesPathStart)
    with open( output_file_name , 'a+') as output:
        output.write(question + "\t" + ques_exact_id + "\n")
        output.write("Gold Answer : "+ " | ".join(answer) + "\n")
        output.write(" \n ".join(map(str, tags_list)))
        output.write("\n")
        
    """
    # Filter questions based on type
    
    spacy_tags_ques_type_list = map_qid_qtype_pair[ques_exact_id].split(" | ")

    #filtered_tag_list = [tag_items for tag_items in tags_list if tags_list[0] in spacy_tags_ques_type_list]
    
    # filter the tags based on the question type tag 
    filtered_tag_list = []
    for tag_items in tags_list:
        ent, tag = tag_items[0] 
        if(tag in spacy_tags_ques_type_list):
            filtered_tag_list.append(tag_items)
    """
    output_file_name = "results/filtered_named_entities_" + quesType
    with open( output_file_name , 'a+') as output:
        output.write(question + "\t" + ques_exact_id + "\n")
        output.write("Gold Answer : "+ " | ".join(answer) + "\n")
        output.write(" \n ".join(map(str, filtered_tag_list)))
        output.write("\n")
    #print(filtered_tag_list)
    """
    #print("Filtered Tag List : ")
    #print(filtered_tag_list)
    doc_content = []

    # Old Code extracting sentences
    """
    for tag_items in filtered_tag_list:
        ent, tag = tag_items[0]
        
        max_cosine_value = -1
        sentences = []
        str_sent = ""
        #print(doc_content[i])
        for i in range(0, 10): 
            sentences += get_sentence_from_entity((doc_content[i]), (ent), tag)
        #print(sentences)
        for sentence in sentences:
            #print(sentence)
            #print(sentence , cosine(model.encode([str.lower(question)])[0], model.encode([str(sentence)])[0]))
            cosine_value = cosine(model.encode([str.lower(question)])[0], model.encode([str(sentence)])[0])
            if cosine_value > max_cosine_value :
                max_cosine_value = cosine_value
                str_sent = str(sentence)

        print(str_sent, max_cosine_value, ent, tag)
    """

    result_list = []
    # run for top k filtered tag list 
    topK_cut_off = 100
    tag_count = 0

    for tag_items in filtered_tag_list:
        tuple_val = tag_items[0]
        max_cosine_value = -1 
        str_sent = ""
        #print(ent_sentence_map[tuple_val])
        sentence_list = ent_sentence_map[tuple_val]
        for sentence in sentence_list:
            cosine_value = cosine(model.encode([str.lower(question)])[0], model.encode([str.lower(sentence)])[0])
            if cosine_value > max_cosine_value :
                max_cosine_value = cosine_value
                str_sent = str(sentence)

        if max_cosine_value != -1 :
            # similarity measure : taking doc freq into account, score = 10/doc_freq + max_    cosine_value; don't want to rerank change this, 2-3 lines
            doc_freq = tag_items[1]
            doc_number = 10    # number of document present is 10 
            #score_tag_tuple = (max_cosine_value,tuple_val)    # earlier sim score
            score_tag_tuple = ((doc_freq * doc_number) * max_cosine_value, tuple_val)
            result_list.append(score_tag_tuple)
        #print(str_sent, max_cosine_value, tuple_val)  # print the max score of sentence and tuples
        
        tag_count += 1
        if tag_count >= topK_cut_off:  # run for top k entities, change topk_cut_off if we want to include more
            break
    result_list = sorted(result_list, key=lambda x: x[0], reverse = True) # sort the list based on cosine values
    
    #exit(1)

    rank_map = {}
    top_scored_result = []
    temp = []

    for tag_items in result_list:
        value = tag_items[0]
        if value not in rank_map:
            rank_map[value] = 1
            if len(temp) != 0:
                top_scored_result.append(" | ".join(temp))
                temp.clear()
        if len(rank_map) >5:
            break
    
        ent, tag = tag_items[1]
        temp.append(str(ent))
    
    result_file_qid_answers[ques_exact_id] = top_scored_result
    #print((result_file_qid_answers))
    #exit(1)
    if i %10 == 0:
        print("Processed %d queries" % i)

    """
    top_scored_result = []
    rank = 1
    rank_map = {}
    temp = []

    for tag_items in filtered_tag_list:
        value = tag_items[1]
        #print(value)
        #exit(1)    
        
        if value not in rank_map :
            rank_map[value] = 1
            if len(temp)!= 0 : 
                top_scored_result.append(" | ".join(temp))
                temp.clear()

        if len(rank_map) > 5 :
            break

        ent, tag = tag_items[0] 
        temp.append(str.lower(ent))
        
    result_file_qid_answers[ques_exact_id] = top_scored_result
    #print(len(result_file_qid_answers))
    #exit(1)    
        # get the only top scored named entities
"""    
    break
# Remaining entities rank them based on some order
json_object = json.dumps(result_file_qid_answers, indent = 4)


with open(resultPathStart + samplingType + "_part_" + quesType + ".json" , "w+") as outfile: 
    outfile.write(json_object)
# TODO : incorporate context while ranking
