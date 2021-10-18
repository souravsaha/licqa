import json

# create the question path
def get_question_path(qid, quesType, quesPathStart):
    ques_path = quesPathStart + "CQ-" + quesType + "_QA.json"
    return ques_path

# create the corpus path
def get_corpus_path(qid, quesType, copusPathStart, samplingType):
    corpus_path = copusPathStart + "CQ_" + quesType + "/" + \
        samplingType + "/" + "CQ-" + quesType + "150-q" + qid + ".json"
    return corpus_path

# create the corpus path
def get_result_path(quesType, resultPathStart, samplingType):
    result_path = resultPathStart + "CQ-" + quesType + "/" + \
        samplingType + "/" + "QUEST_top5_Answers.json"
    return result_path

def get_dr_qa_result_path(quesType, resultPathStart, samplingType):
    result_path = resultPathStart + "CQ-" + quesType + "/" + \
        samplingType + "/" + "DrQA_top5_Answers.json"
    return result_path


# get the question for the particular qid
def get_question_and_gold_answer(qid, quesType, quesPathStart):
    ques_path = get_question_path(qid, quesType, quesPathStart)
    #print(ques_path)
    with open(ques_path) as ques_path_file:
        ques_content = json.load(ques_path_file)
        #print("Quesion : ", ques_content[int(qid) - 1]['question'])
        return ques_content[int(qid) - 1]['question'],ques_content[int(qid) - 1]['gold_answer'], ques_content[int(qid) - 1]['id'] 
