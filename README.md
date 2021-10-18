# LiCQA
This is the code base of our paper on LiCQA : A Lightweight Complex Question Answering System

## Installation
Install LiCQA requirements
```
pip3 install -r requirements.txt
```
## Dataset download 

Download the question sets and corpora from https://quest.mpi-inf.mpg.de/. Create a folder with name 'Data' and place all the corpora and QA files there. 
Download the TREC question type classification data from https://cogcomp.seas.upenn.edu/Data/QA/QC/. Use Training set 5(5500 labeled questions) for training purpose. 



## Running the code
```
Run get_answer_types_from_question.py to obtain the type of the questions.
1. python3 get_answer_types_from_question.py
Run sentence_embedding_ranking.py to get the top k answers.
2. python3 sentence_embedding_ranking.py
Run evalution.py and najork_evaluation.py for vanilla and tie-aware evaluation respectively. 
```
