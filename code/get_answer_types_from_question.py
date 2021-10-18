# TREC Question classifier
# Dataset : https://cogcomp.seas.upenn.edu/Data/QA/QC/
# Report : https://nlp.stanford.edu/courses/cs224n/2010/reports/olalerew.pdf
# Method: Used SVM to classify the questions

# Code: https://github.com/amankedia/Question-Classification/blob/master/Question%20Classifier.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, nltk
import gensim
import codecs
from sner import Ner
import spacy
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
import spacy
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import fbeta_score, accuracy_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer

f_train = open('../trec_qa_train', 'r+')
f_test = open('../trec_qa_valid', 'r+')

train = pd.DataFrame(f_train.readlines(), columns = ['Question'])
test = pd.DataFrame(f_test.readlines(), columns = ['Question'])


train['QType'] = train.Question.apply(lambda x: x.split(' ', 1)[0])
train['Question'] = train.Question.apply(lambda x: x.split(' ', 1)[1])
train['QType-Coarse'] = train.QType.apply(lambda x: x.split(':')[0])
train['QType-Fine'] = train.QType.apply(lambda x: x.split(':')[1])
test['QType'] = test.Question.apply(lambda x: x.split(' ', 1)[0])
test['Question'] = test.Question.apply(lambda x: x.split(' ', 1)[1])
test['QType-Coarse'] = test.QType.apply(lambda x: x.split(':')[0])
test['QType-Fine'] = test.QType.apply(lambda x: x.split(':')[1])


train.head()


#print(test.describe())


#print(test.head())

train.append(test).describe()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(pd.Series(train.QType.tolist() + test.QType.tolist()).values)
train['QType'] = le.transform(train.QType.values)
test['QType'] = le.transform(test.QType.values)
le2 = LabelEncoder()
le2.fit(pd.Series(train['QType-Coarse'].tolist() + test['QType-Coarse'].tolist()).values)
train['QType-Coarse'] = le2.transform(train['QType-Coarse'].values)
test['QType-Coarse'] = le2.transform(test['QType-Coarse'].values)
le3 = LabelEncoder()
le3.fit(pd.Series(train['QType-Fine'].tolist() + test['QType-Fine'].tolist()).values)
train['QType-Fine'] = le3.transform(train['QType-Fine'].values)
test['QType-Fine'] = le3.transform(test['QType-Fine'].values)

all_corpus = pd.Series(train.Question.tolist() + test.Question.tolist()).astype(str)


nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# dot_words = []
# for row in all_corpus:
#     for word in row.split():
#         if '.' in word and len(word)>2:
#             dot_words.append(word)


def text_clean(corpus, keep_list):
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
    
    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process
    
    Output : Returns the cleaned text corpus
    
    '''
    cleaned_corpus = pd.Series()
    for row in corpus:
        qs = []
        for word in row.split():
            if word not in keep_list:
                p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                p1 = p1.lower()
                qs.append(p1)
            else : qs.append(word)
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
    return cleaned_corpus


def preprocess(corpus, keep_list, cleaning = True, stemming = False, stem_type = None, lemmatization = False, remove_stopwords = True):
    
    '''
    Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)
    
    Input : 
    'corpus' - Text corpus on which pre-processing tasks will be performed
    'keep_list' - List of words to be retained during cleaning process
    'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should 
                                                                  be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                  Stemmer. 'snowball' corresponds to Snowball Stemmer
    
    Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together
    
    Output : Returns the processed text corpus
    
    '''
    if cleaning == True:
        corpus = text_clean(corpus, keep_list)
    
    if remove_stopwords == True:
        wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
        stop = set(stopwords.words('english'))
        for word in wh_words:
            stop.remove(word)
        corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    else :
        corpus = [[x for x in x.split()] for x in corpus]
    
    if lemmatization == True:
        lem = WordNetLemmatizer()
        corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    
    if stemming == True:
        if stem_type == 'snowball':
            stemmer = SnowballStemmer(language = 'english')
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
        else :
            stemmer = PorterStemmer()
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    
    corpus = [' '.join(x) for x in corpus]
        

    return corpus


common_dot_words = ['U.S.', 'St.', 'Mr.', 'Mrs.', 'D.C.']
all_corpus = preprocess(all_corpus, keep_list = common_dot_words, remove_stopwords = True)

train_corpus = all_corpus[0:train.shape[0]]
test_corpus = all_corpus[train.shape[0]:]

nlp = spacy.load('en_core_web_sm')

all_ner = []
all_lemma = []
all_tag = []
all_dep = []
all_shape = []
for row in train_corpus:
    doc = nlp(row)
    present_lemma = []
    present_tag = []
    present_dep = []
    present_shape = []
    present_ner = []
    #print(row)
    for token in doc:
        present_lemma.append(token.lemma_)
        present_tag.append(token.tag_)
        #print(present_tag)
        present_dep.append(token.dep_)
        present_shape.append(token.shape_)
    all_lemma.append(" ".join(present_lemma))
    all_tag.append(" ".join(present_tag))
    all_dep.append(" ".join(present_dep))
    all_shape.append(" ".join(present_shape))
    for ent in doc.ents:
        present_ner.append(ent.label_)
    all_ner.append(" ".join(present_ner))

count_vec_ner = CountVectorizer(ngram_range=(1, 2)).fit(all_ner)
ner_ft = count_vec_ner.transform(all_ner)
count_vec_lemma = CountVectorizer(ngram_range=(1, 2)).fit(all_lemma)
lemma_ft = count_vec_lemma.transform(all_lemma)
count_vec_tag = CountVectorizer(ngram_range=(1, 2)).fit(all_tag)
tag_ft = count_vec_tag.transform(all_tag)
count_vec_dep = CountVectorizer(ngram_range=(1, 2)).fit(all_dep)
dep_ft = count_vec_dep.transform(all_dep)
count_vec_shape = CountVectorizer(ngram_range=(1, 2)).fit(all_shape)
shape_ft = count_vec_shape.transform(all_shape)


#x_all_ft_train = hstack([ner_ft, lemma_ft, tag_ft, dep_ft, shape_ft])
x_all_ft_train = hstack([ner_ft, lemma_ft, tag_ft])

# DEBUG :
print("++++++++++++++++++++++ NER ++++++++++++++++++++++")
print(ner_ft[0])
print("-----")
print(all_ner[0])

print("++++++++++++++++++++++ LEMMA +++++++++++++++++++++")
print(lemma_ft[0])
print("-----------")
print(all_lemma[0])

print("++++++++++++++++++++++ TAG +++++++++++++++++++++++")
print(tag_ft[0])
print("------------")
print(all_tag[0])
exit(1)
x_all_ft_train = x_all_ft_train.tocsr()


all_test_ner = []
all_test_lemma = []
all_test_tag = []
all_test_dep = []
all_test_shape = []
for row in test_corpus:
    doc = nlp(row)
    present_lemma = []
    present_tag = []
    present_dep = []
    present_shape = []
    present_ner = []
    #print(row)
    for token in doc:
        present_lemma.append(token.lemma_)
        present_tag.append(token.tag_)
        #print(present_tag)
        present_dep.append(token.dep_)
        present_shape.append(token.shape_)
    all_test_lemma.append(" ".join(present_lemma))
    all_test_tag.append(" ".join(present_tag))
    all_test_dep.append(" ".join(present_dep))
    all_test_shape.append(" ".join(present_shape))
    for ent in doc.ents:
        present_ner.append(ent.label_)
    all_test_ner.append(" ".join(present_ner))


ner_test_ft = count_vec_ner.transform(all_test_ner)
lemma_test_ft = count_vec_lemma.transform(all_test_lemma)
tag_test_ft = count_vec_tag.transform(all_test_tag)
dep_test_ft = count_vec_dep.transform(all_test_dep)
shape_test_ft = count_vec_shape.transform(all_test_shape)

#x_all_ft_test = hstack([ner_test_ft, lemma_test_ft, tag_test_ft, dep_test_ft, shape_test_ft])
x_all_ft_test = hstack([ner_test_ft, lemma_test_ft, tag_test_ft])

x_all_ft_test = x_all_ft_test.tocsr()

model = svm.LinearSVC()

model.fit(x_all_ft_train, train['QType-Coarse'].values)

preds = model.predict(x_all_ft_test)

accuracy_score(test['QType-Coarse'].values, preds)

model.fit(x_all_ft_train, train['QType'].values)

preds = model.predict(x_all_ft_test)

accuracy_score(test['QType'].values, preds)

model.fit(x_all_ft_train, train['QType-Fine'].values)

preds = model.predict(x_all_ft_test)

#print(model.predict('Which photographer did Jana Kramer play in Christmas in Mississipi?'))
#print(preds) 

#accuracy_score(test['QType-Fine'].values, preds)


"""

For each questions CQ-W and CQ-T 
Generate Type information and write into a file

Output file format:

<QID> <Question> <SpacyTag> <TrecCoarseTag> <TrecFinerTag>
TODO : instead of getting type information one by one do in one go. Use 2D
matrix 

"""

## Write binding from TREC Coarse Entity to SPACY Named Enitites
map_TREC_COARSE_to_SPACY_NE = {}

map_TREC_COARSE_to_SPACY_NE['hum'] = ['PERSON']

map_TREC_COARSE_to_SPACY_NE['desc'] = ['']

map_TREC_COARSE_to_SPACY_NE['loc'] = ['GPE', 'LOC', 'ORG']

map_TREC_COARSE_to_SPACY_NE['enty'] = ['NORP', 'FAC', 'PRODUCT', 'EVENT', 'LANGUAGE', 'LAW', 'WORK_OF_ART']

map_TREC_COARSE_to_SPACY_NE['num'] = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

map_TREC_COARSE_to_SPACY_NE['abbr'] = [''] 

map_TREC_COARSE_to_SPACY_NE['money'] = ['MONEY']

map_TREC_COARSE_to_SPACY_NE['date'] = ['DATE']

#Bug fixed: added coarse grained conversion from gr to ORG
map_TREC_COARSE_to_SPACY_NE['gr'] = ['ORG']
# test with a sample question 

from common import get_question_and_gold_answer
import yaml

config_file_name = 'configure.yml'

# defined it here too
with open(config_file_name) as config_file:
    config_file_values = yaml.load(config_file)

qid = config_file_values["qid"]
quesType = config_file_values["quesType"]
quesPathStart = config_file_values["quesPathStart"]

for i in range(150):
    question, answer, ques_exact_id = get_question_and_gold_answer(
        int(qid) +i, quesType, quesPathStart)

    ques_str = question
    # ques_str = "Who played for Barcelona and managed Real Madrid?"

    ques = pd.Series(ques_str).astype(str)
    test_corpus = preprocess(ques, keep_list = common_dot_words, remove_stopwords = True)

    all_test_ner = []
    all_test_lemma = []
    all_test_tag = []
    all_test_dep = []
    all_test_shape = []
    for row in test_corpus:
        doc = nlp(row)
        present_lemma = []
        present_tag = []
        present_dep = []
        present_shape = []
        present_ner = []
        #print(row)
        for token in doc:
            present_lemma.append(token.lemma_)
            present_tag.append(token.tag_)
            #print(present_tag)
            present_dep.append(token.dep_)
            present_shape.append(token.shape_)
        all_test_lemma.append(" ".join(present_lemma))
        all_test_tag.append(" ".join(present_tag))
        all_test_dep.append(" ".join(present_dep))
        all_test_shape.append(" ".join(present_shape))
        for ent in doc.ents:
            present_ner.append(ent.label_)
        all_test_ner.append(" ".join(present_ner))

    ner_test_ft = count_vec_ner.transform(all_test_ner)
    lemma_test_ft = count_vec_lemma.transform(all_test_lemma)
    tag_test_ft = count_vec_tag.transform(all_test_tag)
    dep_test_ft = count_vec_dep.transform(all_test_dep)
    shape_test_ft = count_vec_shape.transform(all_test_shape)

    x_all_ft_test = hstack([ner_test_ft, lemma_test_ft, tag_test_ft])
    x_all_ft_test = x_all_ft_test.tocsr()

    #model.fit(x_all_ft_train, train['QType-Fine'].values)
    test_label = model.predict(x_all_ft_test)

    print(test_label)
    ques_type = list(le3.inverse_transform(test_label))[0]
    print(ques_type)

    f_train = open('../trec_qa_train', 'r+')
    train = pd.DataFrame(f_train.readlines(), columns = ['Question'])

    train['QType'] = train.Question.apply(lambda x: x.split(' ', 1)[0])
    #train['QType-Coarse'] = train.QType.apply(lambda x: x.split(':')[0])
    #train['QType-Fine'] = train.QType.apply(lambda x: x.split(':')[1])

    map_qType_fine_to_Coarse = {}

    #print(train.head())

    ## Code to get the Coarse entity from the model o/p i.e the fine entity 

    for index, item in train.iterrows():
        qtype_fine = item['QType'].split(':')[1]
        qtype_coarse = item['QType'].split(':')[0]
        #print(qtype_fine)
        if qtype_fine not in map_qType_fine_to_Coarse:
            map_qType_fine_to_Coarse[qtype_fine] = qtype_coarse

    #print(map_qType_fine_to_Coarse)
    print("Coarse Entity: ", map_qType_fine_to_Coarse[ques_type])

    print("Spacy NER Tag: ")

    # map from trec coarse/ fine to spacy ner
    ques_type = ques_type.lower()
    coarse_ques_type = map_qType_fine_to_Coarse[ques_type].lower() 

    # spacy NER type
    spacy_ner_tag = "" 

    if ques_type in map_TREC_COARSE_to_SPACY_NE: 
        spacy_ner_tag = map_TREC_COARSE_to_SPACY_NE[ques_type]
        print(spacy_ner_tag)
            
    elif coarse_ques_type in map_TREC_COARSE_to_SPACY_NE: 
        spacy_ner_tag = map_TREC_COARSE_to_SPACY_NE[coarse_ques_type]
        print(spacy_ner_tag)

    else:
        print("unable to find spacy tag for", coarse_ques_type)
    '''
    output_file_name = "results/quesType_finetune_" + quesType
    with open( output_file_name , 'a+') as output:
        output.write(ques_exact_id + "\t" + question + "\t" + " | ".join(spacy_ner_tag) + "\t" + coarse_ques_type + "\t" +  ques_type + "\n")
    '''
#pprint(tags)
#tags_mod = []
#tags_mod = [(text, label) for text, label in tags if label in spacy_ner_tag ] 
#pprint(Counter(tags_mod))

# get the context from document




