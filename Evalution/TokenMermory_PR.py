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
# 加载各个dataset的ID
CWQID = []
webqspID = []
GrailQAID = []
WebQuestionID = []
with open(f"/back-up/gzy/dataset/VLDB/new25/SEID/Token-CWQ.txt","r") as f:
    for line in f:
        CWQID.append(line.strip())
with open(f"/back-up/gzy/dataset/VLDB/new25/SEID/Token-webqsp.txt","r") as f:
    for line in f:
        webqspID.append(line.strip())
with open(f"/back-up/gzy/dataset/VLDB/new25/SEID/Token-GrailQA.txt","r") as f:
    for line in f:
        GrailQAID.append(line.strip())
with open(f"/back-up/gzy/dataset/VLDB/new25/SEID/Token-WebQuestion.txt","r") as f:
    for line in f:
        WebQuestionID.append(line.strip())
IDdic = {
    "CWQ":CWQID,
    "webqsp":webqspID,
    "GrailQA":GrailQAID,
    "WebQuestion":WebQuestionID
}
def getidlist(dataset):
    txtPath = f"/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/Result/Instance_25_{dataset}_sampleID.txt"
    idlist = []
    with open(txtPath, "r") as f:
        for line in f:
            idlist.append(line.strip())
    return idlist
def MemoryprocessSingleJson(jsonpath,idlist,dataset):
    AverageMemory =0
    averageMemorylist =[]
    PeakMemory = 0
    count = 0
    with open(jsonpath, "r") as f:
        infos = json.load(f)
        for i in range(len(infos["eval_info"])):
            data = infos["eval_info"][i]
            id = str(data["id"])
            if id not in idlist or id not in IDdic[dataset]:
                continue
            AverageMemory += data["memory_gpu"]
            averageMemorylist.append(data["memory_gpu"])
            count += 1
        PeakMemory = max(averageMemorylist)
    print("len(averageMemorylist):",len(averageMemorylist))
    AverageMemory = AverageMemory/count
    return PeakMemory,AverageMemory
def Memory(PATHNUM):
    for llm in llmlist:
        for dataset in datasetlist:
            idlist = getidlist(dataset)
            data = [["Instance",f"PeakMemory",f"AverageMemory"]]
            for model in modelist:
                for prtype,result in resultlist.items():
                    modelpath = f"/back-up/gzy/dataset/VLDB/new25/PathRetrieval/{dataset}/{model}"
                    jsonpath = f"{modelpath}/{resultlist[prtype]}.json"
                    PeakMemory,AverageMemory = MemoryprocessSingleJson(jsonpath,idlist,dataset)
                    beforename = namedic[model]+"+"+prtype
                    aftername = newnamedic[beforename]
                    data.append([aftername,PeakMemory,AverageMemory])
            data = sorted(data,key=lambda x:newnamelist.index(x[0]))
            df = pd.DataFrame(data[1:], columns=data[0])
            df.to_excel(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-Memory@{PATHNUM}_PR.xlsx",index=False)
            print(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-Memory@{PATHNUM}_PR.xlsx")
def TokenprocessSingleJson(jsonpath,idlist,dataset):
    EEMsToken =0
    LLMsToken = 0
    count = 0
    with open(jsonpath, "r") as f:
        infos = json.load(f)
        for i in range(len(infos["eval_info"])):
            data = infos["eval_info"][i]
            id = str(data["id"])
            if id not in idlist or id not in IDdic[dataset]:
                continue
            EEMsToken += data["st_tokens"]
            LLMsToken += (data["output_tokens"]+data["input_tokens"])
            count += 1
    EEMsToken = EEMsToken/count
    LLMsToken = LLMsToken/count
    if LLMsToken == 0:
        return EEMsToken,LLMsToken
    else:
        return 0,LLMsToken

def Token(PATHNUM):
    for llm in llmlist:
        for dataset in datasetlist:
            idlist = getidlist(dataset)
            data = [["Instance",f"EEMs Token",f"LLMs Token"]]
            for model in modelist:
                for prtype,result in resultlist.items():
                    modelpath = f"/back-up/gzy/dataset/VLDB/new25/PathRetrieval/{dataset}/{model}"
                    jsonpath = f"{modelpath}/{resultlist[prtype]}.json"
                    EEMsToken,LLMsToken = TokenprocessSingleJson(jsonpath,idlist,dataset)
                    beforename = namedic[model]+"+"+prtype
                    aftername = newnamedic[beforename]
                    data.append([aftername,EEMsToken,LLMsToken])
            data = sorted(data,key=lambda x:newnamelist.index(x[0]))
            df = pd.DataFrame(data[1:], columns=data[0])
            df.to_excel(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-Token@{PATHNUM}_PR.xlsx",index=False)
            print(f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-Token@{PATHNUM}_PR.xlsx")
if __name__ == "__main__":
    llmlist = ["qwen2-7b"]
    datasetlist = ["webqsp","CWQ","GrailQA","WebQuestion"]
    # datasetlist = ["GrailQA"]

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
    # modelist = ["PPR","EMB/edge"]
    PATHNUMlist = [32]
    PATHNUM = 32
    # token
    print("=============开始处理Token================")
    Token(PATHNUM)
    print("==================Token Done==================")
    # memory
    print("=============开始处理Memory================")
    Memory(PATHNUM)
    print("==================Memory Done==================")