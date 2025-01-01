import json
import os
import pandas as pd
from evalut import eval_hr_topk
from evalut import eval_f1
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from MergeGeneration import MergeGeneration
# 写一个字典，把原来的方法映射到论文中的名称
newnamedic ={
    "PPR+GSR":"SBE-PPR+SBR-SPR",
    "PPR+OSR-EEMS":"SBE-PPR+OSAR-EEMs",
    "PPR+OSR-LLMS":"SBE-PPR+OSAR-LLMs",
    "PPR+ISR-EEMs":"SBE-PPR+ISAR-EEMs",
    "PPR+ISR-LLMs":"SBE-PPR+ISAR-LLMs",
    "EMB-edge+GSR":"SAE-EEMs+SBR-SPR",
    "EMB-edge+OSR-EEMS":"SAE-EEMs+OSAR-EEMs",
    "EMB-edge+OSR-LLMS":"SAE-EEMs+OSAR-LLMs",
    "EMB-edge+ISR-EEMs":"SAE-EEMs+ISAR-EEMs",
    "EMB-edge+ISR-LLMs":"SAE-EEMs+ISAR-LLMs",
    "LLM-EMB-edge+GSR":"SAE-LLMs+SBR-SPR",
    "LLM-EMB-edge+OSR-EEMS":"SAE-LLMs+OSAR-EEMs",
    "LLM-EMB-edge+OSR-LLMS":"SAE-LLMs+OSAR-LLMs",
    "LLM-EMB-edge+ISR-EEMs":"SAE-LLMs+ISAR-EEMs",
    "LLM-EMB-edge+ISR-LLMs":"SAE-LLMs+ISAR-LLMs",
}
newnamelist = [
    "Instance",
    "SBE-PPR+SBR-SPR",

    "SAE-EEMs+OSAR-EEMs",
    "SAE-EEMs+OSAR-LLMs",
    "SAE-LLMs+OSAR-EEMs",
    "SAE-LLMs+OSAR-LLMs",
    "SAE-EEMs+ISAR-EEMs",
    "SAE-EEMs+ISAR-LLMs",
    "SAE-LLMs+ISAR-EEMs",
    "SAE-LLMs+ISAR-LLMs",

    "SAE-EEMs+SBR-SPR",
    "SAE-LLMs+SBR-SPR",

    "SBE-PPR+OSAR-EEMs",
    "SBE-PPR+OSAR-LLMs",
    "SBE-PPR+ISAR-EEMs",
    "SBE-PPR+ISAR-LLMs",
]
def getidlist(dataset):
    txtPath = f"/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/Result/Instance_1000_{dataset}_sampleID.txt"
    idlist = []
    with open(txtPath, "r") as f:
        for line in f:
            idlist.append(line.strip())
    return idlist
def PRprocessSingleJson(jsonpath,idlist):
    # 需要算出F1@PATHNUM和Hit@PATHNUM
    Time1 =0
    Time2 = 0
    count = 0
    with open(jsonpath, "r") as f:
        infos = json.load(f)
        for i in range(len(infos["eval_info"])):
            data = infos["eval_info"][i]
            # answers = data["answers"]
            id = str(data["id"])
            if id not in idlist:
                continue
            Time1 += data["structureMethodRetrievalModuleTime"]
            Time2 += data["semanticMethodRetrievalModuleTime"]
            count += 1
    Time1 = Time1/count
    Time2 = Time2/count
    return Time1+Time2
def PRTime(PATHNUM):
    for llm in llmlist:
        for dataset in datasetlist:
            idlist = getidlist(dataset)
            data = [["Instance",f"PRTime"]]
            for model in modelist:
                for prtype,result in resultlist.items():
                    modelpath = f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/{dataset}/{model}"
                    jsonpath = f"{modelpath}/{resultlist[prtype]}/{llm}_{PATHNUM}_{shotsnum}_answers.json"
                    PRTime = PRprocessSingleJson(jsonpath,idlist)
                    beforename = namedic[model]+"+"+prtype
                    aftername = newnamedic[beforename]
                    data.append([aftername,PRTime])
                # print(f"{prtype}处理完成！")
            data = sorted(data,key=lambda x:newnamelist.index(x[0]))
            df = pd.DataFrame(data[1:], columns=data[0])
            df.to_excel(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-PRTime.xlsx",index=False)
            print(f"{dataset}-PRTime Excel处理完成！")
def SEprocessJson(jsonpath,idlist):
    # 需要算出F1@PATHNUM和Hit@PATHNUM
    SETime =0
    count = 0
    with open(jsonpath, "r") as f:
        infos = json.load(f)
        for i in range(len(infos["eval_info"])):
            data = infos["eval_info"][i]
            id = str(data["id"])
            if id not in idlist:
                continue
            SETime += data["semanticMethodPreRetrievalModuleTime"]
            count += 1
    SETime = SETime/count
    return SETime
def SETime():
    jsonpathlist = [
         "/back-up/gzy/dataset/VLDB/new25/SubgraphExtraction/reason_dataset/PPR.json",
        "/back-up/gzy/dataset/VLDB/new25/SubgraphExtraction/reason_dataset/EMB/edge.json",
        "/back-up/gzy/dataset/VLDB/new25/SubgraphExtraction/reason_dataset/LLM/qwen2-70b/EMB/ppr_1000_edge_64.json",
    ]
    namedic = {
        "PPR.json":"SBE",
        "edge.json":"SAE-EEMs",
        "ppr_1000_edge_64.json":"SAE-LLMs",
    }
    for dataset in datasetlist:
        data = [["Instance",f"SETime"]]
        idlist = getidlist(dataset)
        for jsonpath in jsonpathlist:
            SEtime = SEprocessJson(jsonpath.replace("reason_dataset",dataset),idlist)
            # SEtime =50
            aftername = namedic[jsonpath.split("/")[-1]]
            data.append([aftername,SEtime])
        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_excel(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-SETime.xlsx",index=False)
        print(f"{dataset}-SETime Excel处理完成！")
def CombineSEPRTime():
    # 合并SETime和PRTime
    for dataset in datasetlist:
        SEdf = pd.read_excel(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-SETime.xlsx")
        # 把SEdf转化为字典
        SEdic = {}
        for i in range(len(SEdf)):
            SEdic[SEdf["Instance"][i]] = SEdf["SETime"][i]
        PRdf = pd.read_excel(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-PRTime.xlsx")
        # 在PRdf中添加一列SETime,并根据Instance name添加SETime
        PRdf["SETime"] = 0
        for i in range(len(PRdf)):
            sename = PRdf["Instance"][i].split("+")[0].replace("SBE-PPR","SBE")
            PRdf["SETime"][i] = SEdic[sename]
        # 把SETime和PRtime列互换位置
        PRdf = PRdf[[col for col in PRdf.columns if col not in ['SETime', 'PRTime']]+['SETime', 'PRTime']]
        PRdf.to_excel(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-Time.xlsx",index=False)
        print(f"{dataset}-Time Excel处理完成！")

if __name__ == "__main__":
    llmlist = ["qwen2-7b"]
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
    resultlist = {
        "GSR":"SPR",
        "OSR-EEMS":"SPR/EMB",
        "OSR-LLMS":"SPR/LLM/qwen2-70b/EMB",
        "ISR-EEMs":"BeamSearch/EMB",
        "ISR-LLMs":"BeamSearch/LLM/qwen2-70b/EMB"
    }
    namedic = {
        "EMB/edge":"EMB-edge",
        "LLM/qwen2-70b/EMB/ppr_1000_edge_64":"LLM-EMB-edge",
        "PPR":"PPR"
    }
    # shotsNUMlist = ["zero-shot","one-shot","few-shot"]
    shotsnum = "zero_shot"
    modelist = ["PPR","EMB/edge","LLM/qwen2-70b/EMB/ppr_1000_edge_64"]
    PATHNUMlist = [32]
    PATHNUM = 32
    SETime()
    PRTime(PATHNUM)
    CombineSEPRTime()