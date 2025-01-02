import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
def Coverrate(answer,subgraphnode):
    anscount =0
    for ans in answer:
        if ans in subgraphnode:
            return 1
    return 0
def GeRrecall(answer,subgraphnode):
    anscount =0
    for ans in answer:
        if ans in subgraphnode:
            anscount += 1
    return anscount/len(answer)
def GetAcc(answer,subgraphnode):
    anscount =0
    for ans in answer:
        if ans in subgraphnode:
            anscount += 1
    return anscount/len(subgraphnode)
def GetF1(answer,subgraphnode):
    acc = GetAcc(answer,subgraphnode)
    recall = GeRrecall(answer,subgraphnode)
    if acc+recall == 0:
        return 0
    return 2*acc*recall/(acc+recall)
def getgraphnode(subgraph):
    nodeset = set()
    for subgraphnode in subgraph:
        nodeset.add(subgraphnode[0])
        nodeset.add(subgraphnode[2])
    return list(nodeset)
def format_func(value, tick_number):
    return r'$2^{%d}$' % int(value)
def singeInstance(dicpath,metric):
    # 计算F1,recall,和coverrate
    count = 0
    NP = 0
    EP = 0
    TP = 0
    NPpath = dicpath.replace("Prune","node")
    EPpath = dicpath.replace("Prune","edge")
    TPpath = dicpath.replace("Prune","triple")
    # np
    with open(NPpath,"r") as f:
        jsonf  = json.load(f)
        for data in jsonf["eval_info"]:
            try:
                answer = data["answers"]
                acc = data["semanticMethodPreRetrievalModuleACC"]
                a = int(acc.split("/")[0])
                b = int(acc.split("/")[1])
                acc = a/b
                recall = a/len(answer)
                f1 = 2*acc*recall/(acc+recall)
                if metric == "F1":
                    NP += f1
                elif metric == "Recall":
                    NP += recall
                else:
                    print("metric error")
                count += 1
            except Exception as e:
                count+=1
    # ep
    NP = NP/count
    # print(NP)
    count = 0
    with open(EPpath,"r") as f:
        jsonf  = json.load(f)
        # print("file open")
        for data in jsonf["eval_info"]:
            try:
                answer = data["answers"]
                acc = data["semanticMethodPreRetrievalModuleACC"]
                a = int(acc.split("/")[0])
                b = int(acc.split("/")[1])
                acc = a/b
                recall = a/len(answer)
                f1 = 2*acc*recall/(acc+recall)
                if metric == "F1":
                    EP += f1
                    # print("add")
                elif metric == "Recall":
                    EP += recall
                else:
                    print("metric error")
                count += 1
            except Exception as e:
                count+=1
    EP = EP/count
    count = 0
    # tp
    with open(TPpath,"r") as f:
        jsonf  = json.load(f)
        for data in jsonf["eval_info"]:
            try:
                answer = data["answers"]
                acc = data["semanticMethodPreRetrievalModuleACC"]
                a = int(acc.split("/")[0])
                b = int(acc.split("/")[1])
                acc = a/b
                recall = a/len(answer)
                f1 = 2*acc*recall/(acc+recall)
                if metric == "F1":
                    TP += f1
                elif metric == "Recall":
                    TP += recall
                else:
                    print("metric error")
                count += 1
            except Exception as e:
                count+=1
    TP = TP/count

    return NP,EP,TP


def getexcel():
    for metric in metriclist:
        Result = [["windows","NP","EP","TP"]]
        for windows in windowslist:
            pytypeResult = [windows]
            # for prtype in prtypelist:
            NP =0
            EP =0
            TP=0
            for dataset in datasetlist:
                dicpath = f"{ROOTPATH}/{dataset}/subgraph/LLM_token_scale/qwen2-70b/EMB/Prune/{windows}.json"
                np,tp,ep =singeInstance(dicpath,metric) #分别得到NP,EP,TP的值
                NP += np
                EP += tp
                TP += ep
            NP = NP/len(datasetlist)
            EP = EP/len(datasetlist)
            TP = TP/len(datasetlist)
            pytypeResult.append(NP)
            pytypeResult.append(EP)
            pytypeResult.append(TP)
            Result.append(pytypeResult)
        df = pd.DataFrame(Result[1:],columns=Result[0])
        df.to_excel(f"{ROOTPATH}/AverageSubgraph/Average-LLMModel-{metric}.xlsx",index=False)
        print(f"{ROOTPATH}/AverageSubgraph/Average-LLMModel-{metric}.xlsx")
def getImage():
    # 画一个一行两列的折线图，每一个折线图代表一个model的F1或者Recall
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1行6列，调整大小以适应
    axes = axes.flatten()  # 将 2D 数组转为 1D 数组，方便访问每个子图
    index = 0
    for metric in metriclist:
        excelpath = f"{ROOTPATH}/AverageSubgraph/Average-LLMModel-{metric}.xlsx"
        df = pd.read_excel(excelpath)
        windows = df["windows"].tolist()
        x = np.log2(windows)
        y1 = df["NP"].tolist()
        y2 = df["EP"].tolist()
        y3 = df["TP"].tolist()
        ax = axes[index]
        index += 1
        ax.plot(x, y1, label='NP', color='#2dabb2', marker='o', linestyle=':', linewidth=10, markersize=20)
        ax.plot(x, y2, label='EP', color='#daab36', marker='s', linestyle='-', linewidth=10, markersize=20)
        ax.plot(x, y3, label='TP', color='#f0552b', marker='*', linestyle='-.', linewidth=10, markersize=20)
        
        # 设置标题、标签和坐标轴的字号
        ax.set_xlabel("Max Token Num. ", fontsize=30, fontweight='bold')
        ax.set_ylabel(f'{metric}', fontsize=40, fontweight='bold')
        ax.tick_params(axis='x', labelsize=30,length=15)
        ax.tick_params(axis='y', labelsize=30)
        
        ax.set_xlim(9.8, 14.2)
        xticks = [10,11,12,13,14]
        xticklabels = [str(i) for i in range(10, 15, 1)]
        ax.set_xticks(xticks, xticklabels)
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    # 统一添加图例
    font_properties = FontProperties(weight='bold', size=30)
    fig.legend(["NP", "EP","TP"], loc='upper center',prop=font_properties, bbox_to_anchor=(0.5, 1), ncol=3, fontsize=40)
    plt.subplots_adjust(wspace=0.25, hspace=0.55)
    plt.subplots_adjust(top=0.8, left=0.1, right=0.95,bottom=0.24)
    plt.savefig(f"{ROOTPATH}/AverageSubgraph/LLMModel.pdf")
    print(f"{ROOTPATH}/AverageSubgraph/LLMModel.pdf save!")
if __name__ == '__main__':
    ROOTPATH = "/back-up/gzy/dataset/VLDB/SE_new/"
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
    windowslist = [1000,2000,4000,8000,16000]
    metriclist = ["F1","Recall"]
    getexcel()
    print("get excel done!")
    getImage()
    print("get image done!")