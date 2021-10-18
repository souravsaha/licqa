#!/usr/bin/env python3
# coding=utf-8

from openie import StanfordOpenIE
import unicodedata
import codecs
from common import get_corpus_path
import json

quesType = 'W'
copusPathStart = "../Data/Corpora/"
quesPathStart = "../Data/QA/"
samplingType = "top10"
qid = "125"

# normalize the text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text


corpus_path = get_corpus_path(qid, quesType, copusPathStart, samplingType)

# read corpus data from file
with open(corpus_path) as f:
    data = json.load(f)

doc_content = []
for i in range(0, 10):
    # doc_name = data[1]['fname']
    doc_content.append(data[i]['text'])
    doc_content[i] = doc_content[i].replace("\\n", " ")
    # unicodedata.normalize('NFKD', doc_content).encode('ascii','ignore')
    # doubt whether to remove this or not
    doc_content[i] = remove_accented_chars(doc_content[i])
    doc_content[i] = doc_content[i].lower()
    # print(doc_content)

# generate SPO triples
with StanfordOpenIE() as client:

    # text = 'what were the names of the movies that john wayne and maureen ohara worked'
    # print('Text: %s.' % text)
    # for triple in client.annotate(text, properties={'timeout': '50000'}):
    #    print('|-', triple)

    #client.generate_graphviz_graph(text, graph_image)
    #print('Graph generated: %s.' % graph_image)

    #triples_corpus = client.annotate(doc_content)
    #print('Corpus: %s [...].' % corpus[0:80])
    #print('Found %s triples in the corpus.' % len(triples_corpus))
    #print('First 3 triples are...')
    # for triple in triples_corpus:
    #    print('|-', triple)

    for i in range(0, 10) : 
        triples_corpus = client.annotate(doc_content[i], properties={'timeout': '50000'})
        print("For Document %d " % i)
        for triple in triples_corpus:
            #if triple['subject'] == "maureen ohara" or triple['object'] == "maureen ohara" \
            #        or triple['subject'] == "john wayne" or triple['object'] == "john wayne":
            print('|-', triple)

    # graph_image = 'graph.jpg'
    # client.generate_graphviz_graph(doc_content, graph_image)
    # print('generated for big corpus...')
