import json
import os
from tqdm import tqdm
import re


def is_ent(tp_str):
    if len(tp_str) < 3:
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        print(tp_str)
        return True
    return False


def find_entity(sparql_str):
    str_lines = sparql_str.split("\n")
    ent_set = set()
    for line in str_lines[1:]:
        if "ns:" not in line:
            continue

        spline = line.strip().split(" ")
        for item in spline:
            ent_str = item[3:].replace("(", "")
            ent_str = ent_str.replace(")", "")
            if is_ent(ent_str):
                ent_set.add(ent_str)

    return ent_set


ent2name = {}
with open("process_data/id2name.txt", "r") as fp:
    for line in fp.readlines():
        split_line = line.strip().split("\t")
        ent2name[split_line[0]] = split_line[2]

cover = 0
uncover = 0


def id2name(ent):
    global cover, uncover
    if ent in ent2name:
        cover += 1
        return ent2name[ent]
    else:
        # print(ent)
        uncover += 1
        return ent

data_folder = "origin/Freebase/CWQ/"
data_file = ["ComplexWebQuestions_train.json",
             "ComplexWebQuestions_test_wans.json", "ComplexWebQuestions_dev.json"]
# all_data = []
output_file = "process_data/CWQ/CWQ_step0.json"
f_out = open(output_file, "w")
for file in data_file:
    filename = os.path.join(data_folder, file)
    with open(filename) as f_in:
        data = json.load(f_in)
        for q_obj in data:
            # question = q_obj['QuestionText']
            ID = q_obj["ID"]
            # print()
            # answer_list = q_obj["answers"]
            answer_list_new = []
            for answer_obj in q_obj["answers"]:
                new_obj = {}
                new_obj["kb_id"] = answer_obj["answer_id"]
                new_obj["text"] = answer_obj["answer"]
                answer_list_new.append(new_obj)
            question = q_obj["question"]
            sparql_str = q_obj["sparql"]
            # print(question)
            ent_set = find_entity(sparql_str)
            ent_list = [
                {"kb_id": ent, "text": id2name(ent)} for ent in ent_set]
            new_obj = {
                "id": ID,
                "answers": answer_list_new,
                "question": question,
                "entities": ent_list,
            }
            f_out.write(json.dumps(new_obj) + "\n")
            # all_data.append(new_obj)
f_out.close()

print(cover, " ", uncover)

