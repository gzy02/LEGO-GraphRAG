import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import random
from matplotlib.font_manager import FontProperties
# 目标：把ROOTPATH下不同数据集的 GSR,OSR-EEMS,OSR-LLMS,ISR-EEMs,ISR-LLMs的结果进行合并，并画柱状图进行可视化
# 步骤：1.获得整理好的execl表
# 2.画柱状图 F1和Hit@32 分别画，总共需要画10张子图，共两张大图
from evalut import eval_hr_topk
from evalut import eval_f1
PATHNUM = 12
ROOTPATH = "/back-up/gzy/dataset/VLDB/new250/PathRetrieval"
def getidlist(dataset):
    txtPath = f"/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/Result/PR_250_{dataset}_sampleID.txt"
    idlist = []
    with open(txtPath, "r") as f:
        for line in f:
            idlist.append(line.strip())
    return idlist
def processSingleJson(jsonpath,prtype,idlist):
    # 需要算出F1和Hit@32
    F1 =0
    Hit32 = 0
    count = 0
    with open(jsonpath, "r") as f:
        infos = json.load(f)
        for i in range(len(infos["eval_info"])):
            data = infos["eval_info"][i]
            answers = data["answers"]
            prediction = data["ReasoningPaths"].split("\n")
            id = str(data["id"])
            if id not in idlist:
                continue
            prediction = prediction[:PATHNUM]
            F1 += eval_f1(prediction,answers)
            Hit32 += eval_hr_topk(prediction,answers,PATHNUM)
            count += 1
    F1 = F1/count
    Hit32 = Hit32/count
    # print(dataset,jsonpath,count)
    return F1, Hit32
def calculate_position_weighted_averages(file_paths , output_file):
    """
    Calculate position-wise weighted averages across four files for numeric columns
    and keep the first column unchanged. Save the results to an Excel file.

    Parameters:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.
        file3 (str): Path to the third file.
        file4 (str): Path to the fourth file.
        output_file (str): Path to save the output Excel file.

    Returns:
        pd.DataFrame: DataFrame containing the position-wise weighted averages.
    """
    # file_paths = [file1, file2]
    dataframes = [pd.read_excel(file_path) for file_path in file_paths]

    for df in dataframes:
        if not dataframes[0].shape == df.shape:
            raise ValueError("All input files must have the same structure (same rows and columns).")

    first_column = dataframes[0].iloc[:, 0]
    stacked_data = np.stack([df.iloc[:, 1:].values for df in dataframes])
    position_weighted_averages = np.mean(stacked_data, axis=0)

    result_df = pd.DataFrame(
        position_weighted_averages,
        columns=dataframes[0].columns[1:],
        index=dataframes[0].index
    )
    result_df.insert(0, dataframes[0].columns[0], first_column)
    result_df.to_excel(output_file, index=False)

    return result_df
def PR(PATHNUM):
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"] #,"WebQuestion","GrailQA"
    modelist = ["EMB/edge","PPR","LLM/qwen2-70b/EMB/ppr_1000_edge_64"]
    resultlist = {
        "GSR":["SPR.json","EPR.json"],
        "OSR-EEMS":["SPR/BM25.json","SPR/EMB.json","SPR/BGE.json"],
        "OSR-LLMS":["SPR/LLM/qwen2-70b/BGE_new.json"],
        "ISR-EEMs":["BeamSearch/BM25.json","BeamSearch/EMB.json","BeamSearch/BGE.json"],
        "ISR-LLMs":["BeamSearch/LLM/qwen2-70b/BGE_v2.json"]
    }
    print("=====================================")
    print("开始处理单个dataset的结果")
    for dataset in datasetlist:
        idlist = getidlist(dataset)
        for prtype,result in resultlist.items():
            data = [["Method",f"F1@{PATHNUM}",f"Hit@{PATHNUM}"]]
            for r in resultlist[prtype]:
                F1 = 0
                Hit32 = 0
                for model in modelist:
                    modelpath = f"{ROOTPATH}/{dataset}/{model}"
                    jsonpath = f"{modelpath}/{r}"
                    try:
                        f1, hit32 = processSingleJson(jsonpath,prtype,idlist= idlist)
                    except Exception as e:
                        print(e)
                        continue
                    F1 += f1
                    Hit32 += hit32
                F1 = F1/len(modelist)
                Hit32 = Hit32/len(modelist)
                data.append([r,F1,Hit32])
            df = pd.DataFrame(data[1:], columns=data[0])
            # ROOTPATH = "/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval"
            df.to_excel(f"{ROOTPATH}/Result/{dataset}-{prtype}-@{PATHNUM}.xlsx",index=False)
            # print(f"Done {dataset}-{prtype}")
    # 合并四个dataset的结果
    prtypelist = ["GSR","OSR-EEMS","OSR-LLMS","ISR-EEMs","ISR-LLMs"] #"OSR-LLMS","ISR-EEMs"

    print("=====================================")
    print("开始合并四个dataset的结果")
    for prtype in prtypelist:
        # 合并四个dataset的结果
        file1 = f"{ROOTPATH}/Result/CWQ-{prtype}-@{PATHNUM}.xlsx"
        file2 = f"{ROOTPATH}/Result/webqsp-{prtype}-@{PATHNUM}.xlsx"
        file3 = f"{ROOTPATH}/Result/GrailQA-{prtype}-@{PATHNUM}.xlsx"
        file4 = f"{ROOTPATH}/Result/WebQuestion-{prtype}-@{PATHNUM}.xlsx"
        file_paths = [file1, file2,file3,file4]
        output_file = f"{ROOTPATH}/Result/Average-{prtype}-@{PATHNUM}.xlsx"
        result = calculate_position_weighted_averages(file_paths,  output_file)
        print(f"Done Average-{prtype}.xlsx")

from matplotlib.gridspec import GridSpec
def getStructImage(rootpath, saveImgpath, metric):
    import pandas as pd
    plt.figure(figsize=(12, 6))
    legenddic = {
        "EPR.json": "EPR",
        "SPR.json": "SPR",
        "SPR/BM25.json": "BM25",
        "SPR/EMB.json": "ST",
        "SPR/BGE.json": "BGE",
        "SPR/Random.json": "Random",
        "SPR/LLM/qwen2-70b/BM25.json": "BM25",
        "SPR/LLM/qwen2-70b/EMB.json": "LLM-ST",
        "SPR/LLM/qwen2-70b/BGE.json": "BGE",
        "SPR/LLM/qwen2-70b/Random.json": "Random",
        "SPR/LLM/llama3-70b/BM25.json": "BM25",
        "SPR/LLM/llama3-70b/EMB.json": "LLM",
        "SPR/LLM/llama3-70b/BGE.json": "BGE",
        "SPR/LLM/llama3-70b/Random.json": "Random",
        "SPR/LLM/qwen2-70b/BGE_new.json": "BGE",
        "BeamSearch/Random.json": "Random",
        "BeamSearch/BM25.json": "BM25",
        "BeamSearch/EMB.json": "ST",
        "BeamSearch/BGE.json": "BGE",
        "BeamSearch/LLM/qwen2-70b/Random.json": "Random",
        "BeamSearch/LLM/qwen2-70b/BM25.json": "BM25",
        "BeamSearch/LLM/qwen2-70b/EMB_v2.json": "LLM-ST",
        "BeamSearch/LLM/qwen2-70b/BGE_v2.json": "LLM",
        "BeamSearch/LLM/llama3-70b/Random.json": "Random",
        "BeamSearch/LLM/llama3-70b/BM25.json": "BM25",
        "BeamSearch/LLM/llama3-70b/EMB.json": "LLM-ST",
        "BeamSearch/LLM/llama3-70b/BGE.json": "LLM"
    }
    colorslist = ['#00FFFF', '#FFF0F5', 
                '#FFEFD5', '#FFB6C1', '#ADD8E6',
                '#FF4500',
                '#FFEFD5', '#FFB6C1', '#ADD8E6',
                '#FF4500',"#FFD700"]
    # 文件路径
    GSRdf = rootpath + f"/Average-GSR-@{PATHNUM}.xlsx"
    OSREEMSdf = rootpath + f"/Average-OSR-EEMS-@{PATHNUM}.xlsx"
    OSRLLMSdf = rootpath + f"/Average-OSR-LLMS-@{PATHNUM}.xlsx"
    ISREEMsdf = rootpath + f"/Average-ISR-EEMs-@{PATHNUM}.xlsx"
    ISRLLMsdf = rootpath + f"/Average-ISR-LLMs-@{PATHNUM}.xlsx"
    # 读取数据
    df_GSR = pd.read_excel(GSRdf)
    df_OSREEMS = pd.read_excel(OSREEMSdf)
    df_OSRLLMS = pd.read_excel(OSRLLMSdf)
    df_ISREEMs = pd.read_excel(ISREEMsdf)
    df_ISRLLMs = pd.read_excel(ISRLLMsdf)
    dflist = [df_GSR, df_OSREEMS, df_OSRLLMS,df_ISREEMs,df_ISRLLMs]
    df = pd.concat(dflist, axis=0, ignore_index=True)
    # print("==========")
    # print(df)
    # print("==========")
    x = np.arange(len(df["Method"]), dtype=float)
    x[2:] += 1 
    x[5:] +=1 
    x[6:] += 1 
    x[9:] += 1 
    plt.axvspan(-1, 1.8, facecolor='purple', alpha=0.03, label='Basic Group')
    plt.axvspan(1.8, x[4] + 1, facecolor='blue', alpha=0.03, label='Subgraph-Extraction Group')
    plt.axvspan(x[4] + 1, x[5] + 1, facecolor='green', alpha=0.03, label='Path-Filtering Group')
    plt.axvspan(x[5] +1, x[8]+1, facecolor='orange', alpha=0.03, label='Path-Filtering Group')
    plt.axvspan(x[8] + 1, x[-1]+1, facecolor='red', alpha=0.03, label='Path-Filtering Group')
    for i in range(len(df["Method"])):
        plt.bar(
            x[i], df[f'F1@{PATHNUM}'][i], width=0.8, 
            color=colorslist[i], edgecolor='black', 
            linewidth=4, alpha=1
        )
    # plt.bar(df['Method'], df[f'F1@{PATHNUM}'], label='F1', alpha=0.7,color= colorslist)
    #     # 隐藏 x 轴刻度
    x_labels = ["SBR", "OSAR-EEMS", "OSAR-LLMs", "ISAR-EEMS", "ISAR-LLMs"]
    # plt.xticks([])


    plt.yticks(fontsize=40)
    # 设置y轴范围
    plt.ylim(0.1, 0.48)
    plt.yticks(np.arange(0.1, 0.41, 0.1))
    plt.xticks([0.5,4,7,10,13], x_labels, fontsize=40,rotation=15, fontweight='bold')
    # plt.tick_params(axis='x', bottom=False)  # 禁用 x 轴刻度延长线
    plt.tick_params(axis='x', which='major', length=10,labelsize=30)
    # 添加图例
    colorslist = [colorslist[0],colorslist[1],colorslist[2],colorslist[3],colorslist[4],colorslist[5],colorslist[-1]]
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colorslist]
    labels = [legenddic.get(method, method) for method in df['Method']]
    # labels = list(set(labels))
    # print(labels)
    labels = [labels[0],labels[1],labels[2],labels[3],labels[4],labels[-1]]
    font_properties = FontProperties(weight='bold', size=25)
    # 自由设置位置
    plt.legend(handles, labels,loc='center',bbox_to_anchor=(0.49, 0.84), prop=font_properties, ncol=3)
    # plt.ylabel('F1', fontsize=14, fontweight='bold')
    plt.ylabel('F1', fontsize=60, fontweight='bold')
    plt.subplots_adjust(top=0.98,left=0.2,right=0.98,bottom=0.2)
    # plt.tight_layout()
    plt.savefig(saveImgpath)
    print(f"Done {saveImgpath}")


PATHNUMlist = [1,4,8,16,32]
for PATHNUM in PATHNUMlist:
    PR(PATHNUM)
    getStructImage(ROOTPATH+"/Result", ROOTPATH+f"/Result/PR-F1@{PATHNUM}.pdf", f"F1@{PATHNUM}")
    getStructImage(ROOTPATH+"/Result", ROOTPATH+f"/Result/PR-Hit@{PATHNUM}.pdf", f"Hit@{PATHNUM}")

