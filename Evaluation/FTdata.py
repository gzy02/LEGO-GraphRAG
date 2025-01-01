# /home/lzy/expResult/NewResult/CWQ/PPR2/Dij/Emb_FT/paths-PPR2-Dij-Emb_FT.json
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import string
import re
from matplotlib.font_manager import FontProperties

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1  # or s1 in s2


def eval_acc(predictions, answers):
    if len(predictions) == 0:
        return 0
    matched = 0
    for prediction in predictions:
        for ans in answers:
            if match(prediction, ans):
                matched += 1
                break
    return matched / len(predictions)


def eval_recall(predictions, answers):
    if len(answers) == 0:
        return 0
    matched = 0
    for ans in answers:
        for prediction in predictions:
            if match(prediction, ans):
                matched += 1
                break
    return matched / len(answers)


def eval_hit(prediction, answers):
    for a in answers:
        if match(prediction, a):
            return 1
    return 0


def eval_hr_topk(predictions, answers, k):
    for prediction in predictions[:k]:
        for a in answers:
            if match(prediction, a):
                return 1
    return 0


def eval_f1(predictions, answers):
    if len(predictions) == 0:
        return 0

    precision = eval_acc(predictions, answers)
    recall = eval_recall(predictions, answers)
    if precision + recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)

def processSingeJson(jsonpath,idlist,PATHNUM):
    with open(jsonpath, 'r') as file:
        data = json.load(file)
    count = 0
    PostRetrievalModuleF1 = 0
    for i in range(len(data["eval_info"])):
        id = str(data["eval_info"][i]["id"])
        prediction = data["eval_info"][i]["ReasoningPaths"].split("\n")
        # print(id)   
        # if id not in idlist:
        #     continue
        # else:
        prediction = prediction[:PATHNUM]
        # PostRetrievalModuleF1 += data["eval_info"][i]["PostRetrievalModuleF1"]
        answers = data["eval_info"][i]["answers"]
        PostRetrievalModuleF1 += eval_f1(prediction,answers)
        count +=1
    print(jsonpath.split("/")[5]," ",jsonpath.split("/")[-1],"count:",count)
    return PostRetrievalModuleF1/count
def getidlist(instancenum,dataset):
    txtPath = f"/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/Result/{instancenum}_{dataset}_sampleID.txt"
    idlist = []
    with open(txtPath, "r") as f:
        for line in f:
            idlist.append(line.strip())
    return idlist
def getexcel(instancenum):
    for jsonname in jsonlist:
        # print(idlist)
            result = [["PATHNUM","F1"]]
            for PATHNUM in PATHNUMlist:
                F1 = 0
                for dataset in datasetlist:
                    idlist = getidlist(instancenum,dataset)
                    prename = jsonname.split("-")[1]
                    rename = jsonname.split("-")[2]
                    postname = jsonname.split("-")[3].split(".")[0]
                    jsonpath = f"{ROOTPATH}/{dataset}/{prename}/{rename}/{postname}/{jsonname}"
                    F1 += processSingeJson(jsonpath,idlist,PATHNUM)
                F1 = F1/len(datasetlist)
                result.append([PATHNUM,F1])
            df = pd.DataFrame(result[1:],columns=result[0])
            instancename = instancenum.split("_")[1]
            df.to_excel(f"{SAVEPATH}/Average-{legenddic[jsonname]}-{instancename}.xlsx",index=False)
            print(f"{SAVEPATH}/Average-{legenddic[jsonname]}-{instancename}.xlsx save!")
def getImage(instancenum):
    for dataset in datasetlist:
        instancename = instancenum.split("_")[1]
        plt.figure(figsize=(10, 5))
        markerlist = ['o', 's', 'D', '^', 'v', 'p', 'P', '*', 'X', 'd']
        linelist = ['-', '-.', '-.', ':']
        colorlist = ['#daab36','#f0552b','b','c','m','y','k']
        for i in range(len(jsonlist)):
            jsonname = jsonlist[i]
            # prename = jsonname.split("-")[1]
            # rename = jsonname.split("-")[2]
            # postname = jsonname.split("-")[3].split(".")[0]
            df = pd.read_excel(f"{SAVEPATH}/Average-{legenddic[jsonname]}-{instancename}.xlsx")
            # plt.plot(df["PATHNUM"],df["F1"],label=legenddic[jsonname],linewidth=5,markersize=20,marker=)
            plt.plot(df["PATHNUM"],df["F1"],label=legenddic[jsonname],linewidth=5,markersize=20,marker=markerlist[i],linestyle=linelist[i],color=colorlist[i])
        xticks = PATHNUMlist
        xticklabels = [str(i) for i in xticks]
        plt.xticks(xticks, xticklabels)
        plt.xlabel('Pathnum',fontsize=40,fontweight='bold')
        plt.ylabel('F1',fontsize=50,fontweight='bold')
        # plt.title(f'{dataset} F1 of different methods')
        plt.yticks(fontsize=40)
        plt.xticks(fontsize=40)
        # 添加图例
        font_properties = FontProperties(weight='bold', size=35)
        plt.legend(loc='upper left',bbox_to_anchor=(0.1, 0.35),prop=font_properties,ncol=2)
        plt.subplots_adjust(top=0.95, left=0.22, right=0.98,bottom=0.22)
        # plt.tight_layout()
        plt.savefig(f"{SAVEPATH}/Average-FTData-{instancename}.pdf",format="pdf")
        print(f"{SAVEPATH}/Average-FTData-{instancename}.pdf save!")
        
    # excelpath = f"{SAVEPATH}/WebQuestion-FTData-{instancenum.split('_')[1]}.xlsx"

if __name__ == "__main__":
    datasetlist = ["webqsp","CWQ","GrailQA","WebQuestion"]
    INstancelist = ["PR_250"]
    jsonlist = [
        "paths-PPR2-Dij-Emb.json",
        "paths-PPR2-Dij-Emb_FT.json",
    ]
    legenddic = {
        "paths-PPR2-Dij-Emb.json": "ST",
        "paths-PPR2-Dij-Emb_FT.json": "ST-FT",
    }
    ROOTPATH = "/home/lzy/expResult/NewResult"
    SAVEPATH = "/back-up/gzy/dataset/VLDB/new250/PathRetrieval/Result"
    PATHNUMlist = [1,4,8,16,32]
    for instancenum in INstancelist:
        getexcel(instancenum)

    for instancenum in INstancelist:
        getImage(instancenum)
