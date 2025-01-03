import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import random
from matplotlib.font_manager import FontProperties
from evalut import eval_hr_topk
from evalut import eval_f1
# PATHNUM = 12
ROOTPATH = "/back-up/gzy/dataset/VLDB/new25/PathRetrieval"
def getidlist(dataset):
    txtPath = f"/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/Result/PR_250_{dataset}_sampleID.txt"
    idlist = []
    with open(txtPath, "r") as f:
        for line in f:
            idlist.append(line.strip())
    return idlist
def processSingleJson(jsonpath,prtype,idlist):
    PRTime =0
    count =0
    with open(jsonpath, "r") as f:
        infos = json.load(f)
        for i in range(len(infos["eval_info"])):
            data = infos["eval_info"][i]
            prtime = data["semanticMethodRetrievalModuleTime"]+data["structureMethodRetrievalModuleTime"]
            PRTime += prtime
            count += 1
    PRTime = PRTime/count
    return PRTime
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
def PR():
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"] #,"WebQuestion","GrailQA"
    modelist = ["EMB/edge","PPR","LLM/qwen2-70b/EMB/ppr_1000_edge_64"]
    resultlist = {
        "GSR":["SPR.json","EPR.json"],
        "OSR-EEMS":["SPR/BM25.json","SPR/EMB.json","SPR/BGE.json"],
        # "OSR-LLMS":["SPR/LLM/qwen2-70b/BGE_new.json"],
        "ISR-EEMs":["BeamSearch/BM25.json","BeamSearch/EMB.json","BeamSearch/BGE.json"],
        # "ISR-LLMs":["BeamSearch/LLM/qwen2-70b/BGE_v2.json"]
    }
    print("=====================================")
    print("开始处理单个dataset的结果")
    for dataset in datasetlist:
        idlist = getidlist(dataset)
        for prtype,result in resultlist.items():
            data = [["Method",f"PRtime"]]
            for r in resultlist[prtype]:
                PRtime = 0
                for model in modelist:
                    if "BGE_new" in r or "BGE_v2" in r:
                        ROOTPATH = "/back-up/gzy/dataset/VLDB/Pipeline/Generation"
                    else:
                        ROOTPATH = "/back-up/gzy/dataset/VLDB/new25/PathRetrieval"
                    modelpath = f"{ROOTPATH}/{dataset}/{model}"
                    jsonpath = f"{modelpath}/{r}"
                    try:
                        prtime = processSingleJson(jsonpath,prtype,idlist= idlist)
                        # print(f"{jsonpath} {prtime}")
                    except Exception as e:
                        print(f"{jsonpath}")
                        print(e)
                        continue
                    PRtime += prtime
                PRtime = PRtime/len(modelist)
                data.append([r,PRtime])
            df = pd.DataFrame(data[1:], columns=data[0])
            ROOTPATH = "/back-up/gzy/dataset/VLDB/new25/PathRetrieval"
            df.to_excel(f"{ROOTPATH}/Result/{dataset}-{prtype}-PRTime.xlsx",index=False)
    
    prtypelist = ["GSR","OSR-EEMS","ISR-EEMs"] 
    print("=====================================")
    print("开始合并四个dataset的结果")
    for prtype in prtypelist:
        # 合并四个dataset的结果
        file1 = f"{ROOTPATH}/Result/CWQ-{prtype}-PRTime.xlsx"
        file2 = f"{ROOTPATH}/Result/webqsp-{prtype}-PRTime.xlsx"
        file3 = f"{ROOTPATH}/Result/GrailQA-{prtype}-PRTime.xlsx"
        file4 = f"{ROOTPATH}/Result/WebQuestion-{prtype}-PRTime.xlsx"
        file_paths = [file1, file2,file3,file4]
        output_file = f"{ROOTPATH}/Result/Average-{prtype}-PRTime.xlsx"
        result = calculate_position_weighted_averages(file_paths,  output_file)
        print(f"{ROOTPATH}/Result/Average-{prtype}-PRTime.xlsx")
def getStructImage(rootpath, saveImgpath):
    # import pandas as pd
    plt.figure(figsize=(14, 9))
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
        "BeamSearch/LLM/llama3-70b/BGE.json": "LLM",
        "OSAR-LLMs":"LLM",
        "ISAR-LLMs":"LLM",
    }
    colorslist = ['#00FFFF', '#FFF0F5', 
                '#FFEFD5', '#FFB6C1', '#ADD8E6',
                '#FF4500',
                '#FFEFD5', '#FFB6C1', '#ADD8E6',
                '#FF4500']
    # 文件路径
    GSRdf = rootpath + f"/Average-GSR-PRTime.xlsx"
    OSREEMSdf = rootpath + f"/Average-OSR-EEMS-PRTime.xlsx"
    OSRLLMSdf = rootpath + f"/Average-OSR-LLMS-PRTime.xlsx"
    ISREEMsdf = rootpath + f"/Average-ISR-EEMs-PRTime.xlsx"
    ISRLLMsdf = rootpath + f"/Average-ISR-LLMs-PRTime.xlsx"
    # 读取数据
    df_GSR = pd.read_excel(GSRdf)
    df_OSREEMS = pd.read_excel(OSREEMSdf)
    df_OSRLLMS = pd.read_excel(OSRLLMSdf)
    df_ISREEMs = pd.read_excel(ISREEMsdf)
    df_ISRLLMs = pd.read_excel(ISRLLMsdf)
    dflist = [df_GSR, df_OSREEMS, df_OSRLLMS,df_ISREEMs,df_ISRLLMs]
    # dflist = [df_GSR, df_OSREEMS,df_ISREEMs]
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
    plt.axvspan(x[4] + 1, x[7] + 1, facecolor='orange', alpha=0.03, label='Path-Filtering Group')
    plt.axvspan(x[7] +1, x[8]+1, facecolor='orange', alpha=0.03, label='Path-Filtering Group')
    plt.axvspan(x[8] + 1, x[-1]+1, facecolor='red', alpha=0.03, label='Path-Filtering Group')
    for i in range(len(df["Method"])):
        if df[f'PRtime'][i] == 39.61:
            df[f'PRtime'][i] = 6
        elif df[f'PRtime'][i] == 80.35:
            df[f'PRtime'][i] = 6.9
        plt.bar(
            x[i], df[f'PRtime'][i], width=0.8, 
            color=colorslist[i], edgecolor='black', 
            linewidth=4, alpha=1
        )
        if df[f'PRtime'][i] < 0.01:
            plt.text(
            x[i], df[f'PRtime'][i] + 0.05,  # y 值略高于柱子的顶部
            str(round(df[f'PRtime'][i],3)),            # 显示的值
            ha='center',                      # 水平对齐方式
            va='bottom',                      # 垂直对齐方式，'bottom' 使值显示在柱子上方
            fontsize=15,                      # 字体大小
            color='black'                     # 字体颜色
            )
    x_labels = ["SBR", "OSAR-EEMs", "OSAR-LLMs", "ISAR-EEMs", "ISAR-LLMs"]

    plt.yticks(fontsize=35)
    # 设置y轴范围
    plt.ylim(0, 7)
    plt.xticks([0.5,4,7,10,13], x_labels, fontsize=50,rotation=25, fontweight='bold')
    plt.tick_params(axis='x', which='major', length=10,labelsize=35)
    # 添加图例
    yticks = [i for i in range(0,8,1)]
    yticklabels = ["0","1","2","3","4","10","40","80"]
    plt.yticks(yticks, yticklabels)
    colorslist = [colorslist[0],colorslist[1],colorslist[2],colorslist[3],colorslist[4],colorslist[5],colorslist[-1]]
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colorslist]
    labels = [legenddic.get(method, method) for method in df['Method']]
    # labels = list(set(labels))
    # print(labels)
    labels = [labels[0],labels[1],labels[2],labels[3],labels[4],labels[5]]
    font_properties = FontProperties(weight='bold', size=30)
    plt.legend(handles, labels,loc='center',bbox_to_anchor=(0.49, 1.18), prop=font_properties, ncol=3)
    # plt.ylabel('F1', fontsize=14, fontweight='bold')
    plt.ylabel('Time(s)', fontsize=60, fontweight='bold')
    plt.subplots_adjust(top=0.8,left=0.2,right=0.92,bottom=0.2)
    # plt.tight_layout()
    plt.savefig(saveImgpath)
    print(f"Done {saveImgpath}")

PR()
getStructImage(ROOTPATH+"/Result", ROOTPATH+f"/Result/PR-Time.pdf")
