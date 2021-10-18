import json
import yaml
from common import get_question_and_gold_answer, get_result_path, get_dr_qa_result_path
"""
    Hit@5 results: 
    Takes 2 argument one is input question path and another is 
    answer file found from the model

"""

def p_at_1(gold_answer, model_answers):
    gold_answer_list = []
    model_answer_1 = []
    score = 0

    for answer in gold_answer:
        gold_answer_list += answer.split("|")
    gold_answer_list = [x.strip() for x in gold_answer_list] 

    # fixed bug : if model answer doesn't retrieve any answer at all
    if len(model_answers) !=0 :
        model_answer_1 = [x.strip() for x in model_answers[0].split("|")] 
    #print(model_answer_i)
    if not set(model_answer_1).isdisjoint(gold_answer_list):
        score = 1
    return score


def mrr_value(gold_answer, model_answers):
    gold_answer_list = []
    model_answer_i = []
    score = 0

    for answer in gold_answer:
        gold_answer_list += answer.split("|")
    gold_answer_list = [x.strip() for x in gold_answer_list] 

    for i in range(1, 6):
        if len(model_answers) >=i :
            model_answer_i = [x.strip() for x in model_answers[i-1].split("|")] 
        #print(model_answer_i)
        if not set(model_answer_i).isdisjoint(gold_answer_list):
            score = (score + 1)/i
            break           # once matches with break from the loop
    return score


def hit_at_5(gold_answer, model_answers):
    gold_answer_list = []
    model_answer_list = []
    score = 0

    for answer in gold_answer:
        gold_answer_list += answer.split("|")
    gold_answer_list = [x.strip() for x in gold_answer_list] 

    for model_answer in model_answers:
        model_answer_list += model_answer.split("|")
    model_answer_list = [x.strip() for x in model_answer_list] 

    #print("Gold Answer list: ", gold_answer_list)
    #print("Model Gives output: ", model_answer_list)

    if not set(model_answer_list).isdisjoint(gold_answer_list):
        score = 1
    return score

# find the model output for the particular


def get_model_output(ques_exact_id, quesType, resultPathStart, samplingType):

    #result_path = get_result_path(quesType, resultPathStart, samplingType)
    #result_path = get_dr_qa_result_path(quesType, resultPathStart, samplingType)
    # TODO hack for model output file
    #result_path = resultPathStart + "CQ_"+ quesType + "/"+ samplingType + "/sent-embeddings-score_v3_finetuned_type_" + quesType + ".json" 
    #result_path = resultPathStart + "CQ_"+ quesType + "/"+ samplingType + "/sent-embeddings-score_v3_finetuned_type_lg_spacy_" + quesType + ".json" 
    #result_path = resultPathStart + "CQ_"+ quesType + "/"+ samplingType + "/avg_max_every_doc_score_" + quesType + ".json" 
    #result_path = resultPathStart + "CQ_"+ quesType + "/"+ samplingType + "/max_score_plus_doc_freq_flair_" + quesType + ".json" 
    result_path = resultPathStart + "CQ_"+ quesType + "/"+ samplingType + "/max_score_mul_doc_freq_flair_" + quesType + ".json" 
    #result_path = resultPathStart + "CQ_"+ quesType + "/"+ samplingType + "/"+ samplingType + "_" + quesType + ".json" 
    #result_path = resultPathStart + "CQ_"+ quesType + "/"+ samplingType + "/"+ samplingType + "_part_" + quesType + ".json" 
    #result_path = resultPathStart + "CQ_"+ quesType + "/"+ samplingType + "/"+ samplingType + "_re_rank_2_" + quesType + ".json" 
    #result_path = resultPathStart + "CQ_"+ quesType + "/"+ samplingType + "/"+ samplingType + "_re_rank_2_part_" + quesType + ".json" 
    with open(result_path) as result_file:
        result_file_content = json.load(result_file)
        #result_file_value = json.loads(result_file_content)
        result_key = ques_exact_id
        #print(result_path)

        if result_key in result_file_content:
            return result_key, list(result_file_content[result_key]), 1

        return result_key, [], 0

config_file_name = 'configure.yml'

# defined it here too
with open(config_file_name) as config_file:
    config_file_values = yaml.load(config_file)

qid = config_file_values["qid"]
quesType = config_file_values["quesType"]
quesPathStart = config_file_values["quesPathStart"]
resultPathStart = config_file_values["resultPathStart"]
samplingType = config_file_values["samplingType"]

"""
# AVG score for Hit @5
for i in range(150):

    question, answer, ques_exact_id = get_question_and_gold_answer(
        int(qid) + i, quesType, quesPathStart)
    result_key, result_content, success = get_model_output(
        ques_exact_id, quesType, resultPathStart, samplingType)

    print("Ques_exact_id : ", ques_exact_id)
    print("Question is: ", question)
    #print("Gold Answer: ", answer)
    #print("Model Answer : ", result_content)
    if success:
        score = hit_at_5(answer, result_content)
        total_score += score
    else:
        print("Answer not found for the question:")
        score = 0
    print("Hit @5 score : ", score)
    print("\n \n")

print("Total Score", total_score)
print("Avg Hit @5 score : ", total_score/150)
"""
"""
# avg score for mrr
for i in range(150):
    question, answer, ques_exact_id = get_question_and_gold_answer(
        int(qid) +i, quesType, quesPathStart)
    result_key, result_content, success = get_model_output(
        ques_exact_id, quesType, resultPathStart, samplingType)

    print("Ques_exact_id : ", ques_exact_id)
    print("Question is: ", question)
    print("Gold Answer: ", answer)
    print("Model Answer : ", result_content)
    if success:
        score = mrr_value(answer, result_content)
        total_score += score
    else:
        print("Answer not found for the question:")
        score = 0
    print("MRR score : ", score)
    print("\n \n")

print("Total Score", total_score)
print("Avg mrr score : ", total_score/150)
"""
"""
# avg score for p@1
for i in range(150):
    question, answer, ques_exact_id = get_question_and_gold_answer(
        int(qid) +i, quesType, quesPathStart)
    result_key, result_content, success = get_model_output(
        ques_exact_id, quesType, resultPathStart, samplingType)

    print("Ques_exact_id : ", ques_exact_id)
    print("Question is: ", question)
    print("Gold Answer: ", answer)
    print("Model Answer : ", result_content)
    if success:
        score = p_at_1(answer, result_content)
        total_score += score
    else:
        print("Answer not found for the question:")
        score = 0
    print("P@1 score : ", score)
    print("\n \n")

print("Total Score", total_score)
print("Avg p@1 score : ", total_score/150)
"""
def compute_score(evaluation_type):
    evaluation_measure =""
    total_score = 0
    score = 0

    if evaluation_type == "t":
        evaluation_measure = "Hit@5"
        
    if evaluation_type == "m":
        evaluation_measure = "MRR"
        
    if evaluation_type == "p":
        evaluation_measure = "P@1"
    
    outputfile = open(args.filepath, "a")

    for i in range(150):
        question, answer, ques_exact_id = get_question_and_gold_answer(
            int(qid) +i, quesType, quesPathStart)
        result_key, result_content, success = get_model_output(
            ques_exact_id, quesType, resultPathStart, samplingType)

        # lower case of result list
        result_content = [x.lower() for x in result_content ]

        print("Ques_exact_id : ", ques_exact_id)
        print("Question is: ", question)
        print("Gold Answer: ", answer)
        print("Model Answer : ", result_content)
        if success:
            #score = p_at_1(answer, result_content)
            if evaluation_type == "t":
                score = hit_at_5(answer, result_content)

            elif evaluation_type == "m":
                score = mrr_value(answer, result_content)

            elif evaluation_type == "p":
                score = p_at_1(answer, result_content)
            
            total_score += score
        else:
            print("Answer not found for the question:")
            score = 0
        
        #outputfile.write(evaluation_measure + "\t" + ques_exact_id + "\t" + str(score) + "\n")
        outputfile.write(evaluation_measure + "\t" + str(int(qid) +i) + "\t" + str(score) + "\n")
        print("{} score : {}".format(evaluation_measure,score))
        print("\n \n")

    print("Total Score", total_score)
    print("Avg {} score : {}".format(evaluation_measure,total_score/150))
    outputfile.close()

import argparse
import sys
parser = argparse.ArgumentParser()

# Adding argument
parser.add_argument("-t", "--top", help = "Hit @ 5 measure", action='store_true') 
parser.add_argument("-m", "--mrr", help = "MRR measure", action='store_true') 
parser.add_argument("-p", "--precision", help = "p@1 measure", action='store_true') 
parser.add_argument("-f", "--filepath", help="store outputfile")
# Read arguments from command line

args = parser.parse_args()

if not len(sys.argv) > 1: 
    parser.print_help()
    exit(1)

if args.top:
    avg_score = compute_score("t")
if args.mrr:
    avg_score = compute_score("m")
if args.precision:
    avg_score = compute_score("p")

#filepath = args.filepath
if None == args.filepath : 
    args.filepath = "foo"

#print(args.filepath)
