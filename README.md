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
1. Set question path, question type, corpora path, sampling type, output path in the configure.yml file.
2. Run get_answer_types_from_question.py to obtain the type of the questions.
   python3 get_answer_types_from_question.py
3. Run sentence_embedding_ranking.py to get the top k answers.
   python3 sentence_embedding_ranking.py
4. Run evalution.py and najork_evaluation.py for vanilla and tie-aware evaluation respectively. 
   
```
