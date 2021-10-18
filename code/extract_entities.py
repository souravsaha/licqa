# !/usr/local/bin/python3
from common import get_question_and_gold_answer, get_corpus_path
import yaml
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import json
import unicodedata
import re
from pprint import pprint
from contractions import CONTRACTION_MAP

#import en_core_web_lg
#nlp = en_core_web_lg.load()
nlp = en_core_web_sm.load()
nlp.max_length = 1030000
#nlp.max_length = 1074115

"""
Add coref resolution to the pipeline
"""
#import neuralcoref
#neuralcoref.add_to_pipe(nlp)

#import os.path
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#from get_answer_types_from_question import get_answer_type_main
#from stanfordcorenlp import StanfordCoreNLP

#stanford_nlp = StanfordCoreNLP(r'../Code/stanford-corenlp-full-2018-10-05', quiet=False)

"""
 input : Data file
 Extract the documents from the json for each query, 
 ideally we will have 10 documents
"""


def get_entities(document_content):
    doc = nlp(document_content)
    doclist = list(doc.sents)

    sentences = [x for x in doc.sents]

    #displacy.render(nlp(str(sentences)), jupyter=True, style='ent')
    # pprint(doclist[0])
    # sys.stdin.read(1)
    #tags = [(X.text, X.label_, X.start, X.end) for X in doc.ents]

    tags = [(X.text, X.label_) for X in doc.ents]
    position = [(X.start, X.end) for X in doc.ents]
    return tags

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


"""
Configuration of question and corpus
TODO
put it in configure.yml file  
"""
# question type : T/W
# sampling type : top10/strata1/strata2/strata3/strata4/strata5
# qid : query id of the question


def get_named_entities(qid, quesType, quesPathStart, corpus_path):
    question, answer, ques_exact_id = get_question_and_gold_answer( qid, quesType, quesPathStart)

    print("Question : ", question)
    print("Gold Answer : ", answer)
    # take the file name as parameter
    # corpus path is like : '../Data/Corpora/CQ_W/top10/CQ-W150-q006.json'

    #print(get_answer_type_main(question, stanford_nlp))

    #corpus_path = get_corpus_path(qid, quesType, corpusPathStart, samplingType)
    # print(corpus_path)
    with open(corpus_path) as f:
        data = json.load(f)
    tags = []
    doc_context = []
    for i in range(0, 10):
        doc_name = data[i]['fname']
        doc_content = data[i]['text']
        doc_content = doc_content.replace("\n", " ").strip()
        # convert unicode characters

        #doc_content = unicode(doc_content, "utf-8")
        #doc_content = doc_content.strip().strip('\n').encode('ascii','ignore')
        #print("\nDocument : ", doc_name)

        doc_content = unicodedata.normalize(
            'NFKD', doc_content).encode('ascii', 'ignore')
        doc_content = str(doc_content)
        doc_content = expand_contractions(doc_content)

        tags += list(set(get_entities(doc_content)))
        # pprint(Counter(tags))

    #pprint(Counter(tags))
    return Counter(tags)

    """
    output_file_name = "results/quesType_" + quesType + "entities_order_df"
    with open( output_file_name , 'a+') as output:
        output.write(ques_exact_id + "\t" + question + "\n")
        output.write(Counter(tags) + "\n")
    """
#sentence = "Troy (2004) was an epic film about the Trojan war based on Homer's poem The Iliad. Troy featured a star-studded cast list, including Brad Pitt as Achilles and Diane Kruger as Helen of Troy, the \"face that launched a thousand ships.\" This list of Troy actors includes any Troy actresses and all other actors from the film. You can view trivia about each Troy actor on this list, such as when and where they were born. To find out more about a particular actor or actress, click on their name and you'll be taken to a page with even more details about their acting career. The cast members of Troy have been in many other movies, so use this list as a starting point to find actors or actresses that you may not be familiar with.\nThis list contains actors like Orlando Bloom, Rose Byrne, and the characters they played in Troy.\nIf you want to answer the questions, \"Who starred in the movie Troy?\" and \"What is the full cast list of Troy?\" then this page has got you covered.\n"
# question: which actor played in troy and seven?
# answer : brad pitt | brad | pitt

"""
 TODO
1. wh question classifier

2. df er context -> wip  
    <start, end> => dekhate hobe orom kore  
>> DONE

3. <subject, predicate, object>  

4. ascii kore nite hobe entity guloke/ puro doc takei ascii te felte hobe
>> DONE

5. pre process the documents.. Barça  => barca
>> DONE 

"""
import numpy as np

def get_entities_sent(document_content, ent_sentence_map):
    doc = nlp(document_content)
    doclist = list(doc.sents)

    sentences = [x for x in doc.sents]

    #displacy.render(nlp(str(sentences)), jupyter=True, style='ent')
    # pprint(doclist[0])
    # sys.stdin.read(1)
    #tags = [(X.text, X.label_, X.start, X.end) for X in doc.ents]

    tags = [(X.text, X.label_) for X in doc.ents]
    position = [(X.start, X.end) for X in doc.ents]

    tags = []
    #ent_sentence_map = {}
    
    for sent in doc.sents:
      tmp = nlp(sent.text)
      for ent in tmp.ents:
        temp_tuple = (ent.text, ent.label_)
        tags.append(temp_tuple)

        if temp_tuple not in ent_sentence_map:
          sentences = []
          sentences.append(str(sent))
          ent_sentence_map[temp_tuple] = sentences
        else:
          old = ent_sentence_map[temp_tuple]
          old.append(str(sent))
          ent_sentence_map[temp_tuple] = old
        #print(ent.text, ent.label_, sent)
    
    return tags, ent_sentence_map

from segtok.segmenter import split_single
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load('ner-ontonotes')

# format from [DATE (0.8909)] to DATE
def format_to_onto_label(input_label):

    input_label = input_label.replace("[","")
    input_label = input_label.replace("]","")
    
    return input_label.split(" ")[0]


def get_entities_flair(document_content, ent_sentence_map):
    corpus = [Sentence(sent, use_tokenizer=True) for sent in split_single(str(document_content))]
    #print(sentences)
    tags = []
    tagger.predict(corpus)

    for sent in corpus:
        #print(sent)
        for entity in sent.get_spans('ner'):
            #print(entity.labels, entity.text)
            #exit(0)
            label = format_to_onto_label(str(entity.labels))
            temp_tuple = (str(entity.text), label)

            tags.append(temp_tuple)

            if temp_tuple not in ent_sentence_map:
                sentences = []
                sentences.append(str(sent.to_plain_string()))
                ent_sentence_map[temp_tuple] = sentences
            else:
                old = ent_sentence_map[temp_tuple]
                old.append(str(sent.to_plain_string()))
                ent_sentence_map[temp_tuple] = old

    return tags, ent_sentence_map
    
    
def get_named_entities_with_sentence(qid, quesType, quesPathStart, corpus_path):
    question, answer, ques_exact_id = get_question_and_gold_answer( qid, quesType, quesPathStart)

    print("Question : ", question)
    print("Gold Answer : ", answer)
    # take the file name as parameter
    # corpus path is like : '../Data/Corpora/CQ_W/top10/CQ-W150-q006.json'

    #print(get_answer_type_main(question, stanford_nlp))

    #corpus_path = get_corpus_path(qid, quesType, corpusPathStart, samplingType)
    # print(corpus_path)
    with open(corpus_path) as f:
        data = json.load(f)
    tags = []
    doc_context = []
    ent_sentence_map = {}
    for i in range(0, 10):
        doc_name = data[i]['fname']
        doc_content = data[i]['text']
        doc_content = doc_content.replace("\n", " ").strip()
        # convert unicode characters

        #doc_content = unicode(doc_content, "utf-8")
        #doc_content = doc_content.strip().strip('\n').encode('ascii','ignore')
        #print("\nDocument : ", doc_name)

        doc_content = unicodedata.normalize(
            'NFKD', doc_content).encode('ascii', 'ignore')
        doc_content = str(doc_content)
        doc_content = expand_contractions(doc_content)

        #tags += list(set(get_entities(doc_content)))
        # get_entitites_sent is using spacy NER , we can replace it to check how it's working with other NER
        # Check with coref resolution 
        #import neuralcoref
        #neuralcoref.add_to_pipe(nlp)
        #doc = nlp(doc_content)
        #doc_content = doc._.coref_resolved
        #ent_tags, ent_sentence_map = get_entities_sent(doc_content, ent_sentence_map)
        ent_tags, ent_sentence_map = get_entities_flair(doc_content, ent_sentence_map)
        tags += list(set(ent_tags))
        #ent_sentence_map += ent_sentence_temp

        # pprint(Counter(tags))

    #pprint(Counter(tags))
    return Counter(tags), ent_sentence_map
