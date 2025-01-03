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
from evalut import normalize
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
    "Method",
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
def processSingleJson(jsonpath,dataset,hoptype):
    # 需要算出F1@PATHNUM和Hit@PATHNUM
    F1 =0
    Hit = 0
    count = 0
    with open(jsonpath, "r") as f:
        infos = json.load(f)
        for i in range(len(infos["eval_info"])):
            data = infos["eval_info"][i]
            answers = data["answers"]
            id = str(data["id"])
            question = normalize(data["question"])
            if dataset=="WebQuestion":
                hop = questin2hop[question]
            else:
                hop = id2hop[id]
            if hoptype == "OneHop" and hop == 1:
                F1 += data["F1"]
                Hit += data["HR"]
                count += 1
            elif hoptype == "TwoHop" and hop == 2:
                F1 += data["F1"]
                Hit += data["HR"]
                count += 1
            elif hoptype == "MultiHop" and hop > 1:
                F1 += data["F1"]
                Hit += data["HR"]
                count += 1
    F1 = F1/count
    Hit = Hit/count
    return F1, Hit
def getidlist(dataset,hop):
    txtPath = f"/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/Result/Instance_1000_{dataset}_sampleID.txt"
    idlist = []
    if dataset=="WebQuestion":
        IDgetHop = questin2hop
    with open(txtPath, "r") as f:
        for line in f:
            idhop = line.strip()
            if hop =="SingleHop" and IDgetHop(idhop) == 1:
                idlist.append(idhop)
            elif hop =="MultiHop" and IDgetHop(idhop) > 1:
                idlist.append(idhop)
    return idlist
def generation(PATHNUM,hoptype):
    for llm in llmlist:
        data = [["Method",f"F1",f"HR"]]
        for model in modelist:
            for prtype,result in resultlist.items():
                F1 = 0
                Hit32 = 0
                for dataset in datasetlist:
                    # idlist = getidlist(dataset,hop)
                    # modelpath = f"/back-up/gzy/dataset/VLDB/Instance/NewGeneration/{dataset}/{model}"
                    # jsonpath = f"{modelpath}/{resultlist[prtype]}/{llm}_{PATHNUM}_{shotsnum}_answers.json"
                    # f1, hit32 = processSingleJson(jsonpath,dataset,hoptype)
                    try:
                        modelpath = f"/back-up/gzy/dataset/VLDB/Pipeline/NewGeneration/{dataset}/{model}"
                        jsonpath = f"{modelpath}/{resultlist[prtype]}/{llm}_{PATHNUM}_{shotsnum}_answers.json"
                        f1, hit32 = processSingleJson(jsonpath,dataset,hoptype)
                        print("使用新数据成功！")
                    except:
                        print("新数据暂未推理,使用旧的数据：")
                        modelpath = f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/{dataset}/{model}"
                        jsonpath = f"{modelpath}/{resultlist[prtype]}/{llm}_{PATHNUM}_{shotsnum}_answers.json"
                        f1, hit32 = processSingleJson(jsonpath,dataset,hoptype)
                    F1 += f1
                    Hit32 += hit32
                    beforename = namedic[model]+"+"+prtype
                    aftername = newnamedic[beforename]
                F1 = F1/len(datasetlist)
                Hit32 = Hit32/len(datasetlist)
                data.append([aftername,F1,Hit32])
        data = sorted(data,key=lambda x:newnamelist.index(x[0]))
        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_excel(f"/back-up/gzy/dataset/VLDB/Instance/NewGeneration/Result/Average-{llm}-@{PATHNUM}-{hoptype}.xlsx",index=False)
        print(f"Average-{llm}-@{PATHNUM}-{hoptype}.xlsx Excel处理完成！")
            
            
def GenerationImage(rootpath, saveImgpath):
    custom_colors = [
        "#293844",
        "#FAF0EB","#FCDCC6","#f79767","#f87349","#f64c29","#d32912","#DEBC33","#720909",
        "#42CFFE", "#0487e2",
        "#C8EAD1","#73C088","#397D54", "#3D4F2F"
        # "#293844",
        # "#f4f1d0","#E3E9D7","#C8D7B4","#eae0ab","#B7A368","#739353","#50673A","#3D4F2F",
        # "#75b5dc","#478ecc",
        # "#c87d98","#b25f79", "#9b3f5c", "#832440", 
    ]
    fig, ax = plt.subplots(1, 3, figsize=(30, 5))
    ax = ax.flatten()
    for k in range(0,1):
        for i in range(0,3):
            if i==0:
                hoptype = "OneHop"
                xlabelname = "(a) One-Hop"
            elif i ==1:
                hoptype = "TwoHop"
                xlabelname = "(b) Two-Hop"
            else:
                hoptype = "MultiHop"
                xlabelname = "(c) Multi-Hop"
            llmpath = f"{rootpath}/Average-{llmlist[i]}-@{PATHNUM}-{hoptype}.xlsx"
            df = pd.read_excel(llmpath)
            Hitlist = df[f"HR"].tolist()
            x = np.arange(len(Hitlist), dtype=float)
            x[1:] += 0.5 
            x[9:] += 0.5 
            x[11:] += 0.5 
            x[15:] += 0.5 
            methodlist = df["Method"].tolist()
            # x = np.arange(len(Hitlist))
            # 设置坐标轴刻度，a为最小值，b为最大值
            a = min(Hitlist)-0.1
            b = max(Hitlist)+0.1
            ax[i].set_ylim(0.3, 0.75)
            ax[i].axvspan(-1, 0.7, facecolor='#B2B3B0', alpha=0.1, label='Basic Group')
            ax[i].axvspan(0.7, x[8] + 0.7, facecolor='#fee3ce', alpha=0.3, label='Subgraph-Extraction Group')
            ax[i].axvspan(x[8] + 0.7, x[10] + 0.8, facecolor='#B7A368', alpha=0.1, label='Path-Filtering Group')
            ax[i].axvspan(x[10] + 0.8, x[-1]+1, facecolor='#E3E9D7', alpha=0.3, label='Path-Filtering Group')
            for j in range(len(Hitlist)):
                ax[i].bar(
                    x[j], Hitlist[j], width=0.8, 
                    color=custom_colors[j], edgecolor='black', 
                    linewidth=4, alpha=1
                )
            ax[i].set_xticks(x,)
            ax[i].set_xlabel(xlabelname, fontsize=40, fontweight='bold')
            ax[i].set_ylabel(f"Hits@1", fontsize=40, fontweight='bold')
            # 隐藏 X 轴刻度和标签
            ax[i].tick_params(axis='x', which='both', length=0)
            # ax[i].tick_params(axis='x', which='both', length=0)  # 隐藏刻度短线
            ax[i].set_xticklabels(range(1,16))  # 隐藏 X 轴标签
            ax[i].tick_params(axis='x', which='major', labelsize=30, width=3,length=10,rotation=55)
            ax[i].tick_params(axis='y', which='major', labelsize=30, width=3,length=20)
            ax[i].xaxis.labelpad = 10
            ax[i].text(-0.5, 0.25, "No.", ha='right', va='center', fontsize=30, fontweight='bold') 

    # legend_elements = [Patch(color=color, label=method) for color, method in zip(custom_colors, methodlist)]
    plt.subplots_adjust(wspace=0.25, hspace=0.55)  # 设置行和列之间的间隔
    # font_properties = FontProperties(weight='bold', size=30)
    # plt.title(f"{dataset} Generation", fontsize=60, fontweight='bold')
    #plt.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(-1.85, 3.5), fontsize=30, prop=font_properties,labelspacing=0.05)
    plt.subplots_adjust(top =0.99,left=0.06, right=0.99,bottom=0.3)
    plt.savefig(saveImgpath, format='pdf')
    # plt.tight_layout()
    print("画图保存成功：", saveImgpath)
if __name__ == "__main__":
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
    datasetrootpath = "/home/gzy/graphrag/src/local_data"
    id2hop = {}
    questin2hop = {}
    for dataset in datasetlist:
        datasetpath = f"{datasetrootpath}/{dataset}_hop.json"
        if dataset=="WebQuestion":
            # 根据question找hop
            with open(datasetpath, "r") as f:
                hopinfo = json.load(f)
                for data in hopinfo:
                    question = normalize(data["question"])
                    questin2hop[question] = data["hop"]
        else:
            # 根据id找hop
            with open(datasetpath, "r") as f:
                hopinfo = json.load(f)
                for data in hopinfo:
                    id2hop[str(data["id"])] = data["hop"]
    llmlist = ["qwen2-7b","qwen2-7b","qwen2-7b"]
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
    resultlist = {
        "GSR":"SPR",
        "OSR-EEMS":"SPR/EMB",
        "OSR-LLMS":"SPR/LLM/qwen2-70b/EMB",
        "ISR-EEMs":"BeamSearch/EMB",
        "ISR-LLMs":"BeamSearch/LLM/qwen2-70b/EMB_v2"
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
    for PATHNUM in PATHNUMlist:
        generation(PATHNUM,"OneHop")
        generation(PATHNUM,"TwoHop")
        generation(PATHNUM,"MultiHop")
    GenerationImage(f"/back-up/gzy/dataset/VLDB/Instance/NewGeneration/Result",f"/back-up/gzy/dataset/VLDB/Instance/NewGeneration/Result/Average-Generation-Hit@{PATHNUM}.pdf")