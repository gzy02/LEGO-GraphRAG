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
    txtPath = f"/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/Result/Instance_25_{dataset}_sampleID.txt"
    idlist = []
    with open(txtPath, "r") as f:
        for line in f:
            idlist.append(line.strip())
    return idlist
def TokenprocessSingleJson(jsonpath,idlist,dataset):
    EEMsToken =0
    LLMsToken = 0
    count = 0
    seidlist =[] 
    # 把出现过的id放到idlist中，存到txt文件中
    with open(jsonpath, "r") as f:
        infos = json.load(f)
        for i in range(len(infos["eval_info"])):
            data = infos["eval_info"][i]
            id = str(data["id"])
            if id not in idlist:
                continue
            EEMsToken += data["st_tokens"]
            LLMsToken += (data["output_tokens"]+data["input_tokens"])
            count += 1
            seidlist.append(id)
    # jsonpath = jsonpath.replace("/back-up/gzy/dataset/VLDB/new25/SubgraphExtraction","/back-up/gzy/dataset/VLDB/Pipeline/subgraph/")
    # with open(jsonpath, "r") as f:
    #     infos = json.load(f)
    #     for i in range(len(infos["eval_info"])):
    #         data = infos["eval_info"][i]
    #         id = str(data["id"])
    #         if id not in idlist:
    #             continue
    #         EEMsToken += data["st_tokens"]
    #         LLMsToken += (data["output_tokens"]+data["input_tokens"])
    #         count += 1
    #         seidlist.append(id)
    with open(f"/back-up/gzy/dataset/VLDB/new25/SEID/Token-{dataset}.txt","w") as f:
        for id in seidlist:
            f.write(id+"\n")
    EEMsToken = EEMsToken/count
    LLMsToken = LLMsToken/count
    if LLMsToken == 0:
        return EEMsToken,LLMsToken
    else:
        return 0,LLMsToken
def Token(PATHNUM):
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
        data = [["Instance",f"EEMs Token",f"LLMs Token"]]
        idlist = getidlist(dataset)
        for jsonpath in jsonpathlist:
            EEmsToken,LLMsToken = TokenprocessSingleJson(jsonpath.replace("reason_dataset",dataset),idlist,dataset)
            aftername = namedic[jsonpath.split("/")[-1]]
            data.append([aftername,EEmsToken,LLMsToken])
        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_excel(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-Token@{PATHNUM}_SE.xlsx",index=False)
        print(f"//back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-Token@{PATHNUM}_SE.xlsx")
def MemoryprocessSingleJson(jsonpath,idlist,dataset):
    AverageMemory =0
    averageMemorylist =[]
    PeakMemory = 0
    count = 0
    seidlist =[]
    with open(jsonpath, "r") as f:
        infos = json.load(f)
        for i in range(len(infos["eval_info"])):
            data = infos["eval_info"][i]
            id = str(data["id"])
            if id not in idlist:
                continue
            AverageMemory += data["memory_gpu"]
            averageMemorylist.append(data["memory_gpu"])
            count += 1
            seidlist.append(id)
        PeakMemory = max(averageMemorylist)
    AverageMemory = AverageMemory/count
    with open(f"/back-up/gzy/dataset/VLDB/new25/SEID/Memory-{dataset}.txt","w") as f:
        for id in seidlist:
            f.write(id+"\n")
    return PeakMemory,AverageMemory
def Memory(PATHNUM):
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
        data = [["Instance",f"PeakMemory",f"AverageMemory"]]
        idlist = getidlist(dataset)
        for jsonpath in jsonpathlist:
            PeakMemory,AverageMemory = MemoryprocessSingleJson(jsonpath.replace("reason_dataset",dataset),idlist,dataset)
            aftername = namedic[jsonpath.split("/")[-1]]
            data.append([aftername,PeakMemory,AverageMemory])
        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_excel(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-Memory@{PATHNUM}_SE.xlsx",index=False)
        print(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-Memory@{PATHNUM}_SE.xlsx")
if __name__ == "__main__":
    datasetlist = ["webqsp","CWQ","GrailQA","WebQuestion"]
    # datasetlist = ["GrailQA"]

    PATHNUM = 32
    # token
    print("=============开始处理Token================")
    Token(PATHNUM)
    print("==================Token Done==================")
    # memory
    print("=============开始处理Memory================")
    Memory(PATHNUM)
    print("==================Memory Done==================")