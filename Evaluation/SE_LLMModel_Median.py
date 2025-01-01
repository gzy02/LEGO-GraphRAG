import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
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
def get_median(numbers):
    # 首先排序列表
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    # 如果列表长度是奇数，返回中间的数字
    if n % 2 != 0:
        return sorted_numbers[n // 2]
    # 如果列表长度是偶数，返回中间两个数字的平均值
    else:
        middle1 = sorted_numbers[n // 2 - 1]
        middle2 = sorted_numbers[n // 2]
        return (middle1 + middle2) / 2
def singeInstance(dicpath,metric):
    # 计算F1,recall,和coverrate
    count = 0
    NP = 0
    EP = 0
    TP = 0

    # 得到中位数
    NPlist = []
    EPlist = []
    TPlist = []
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
                triplenum = ast.literal_eval(data["afterSemanticMethodPreRetrievalModule"])["edges"]
                NPlist.append(triplenum)
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
                triplenum = ast.literal_eval(data["afterSemanticMethodPreRetrievalModule"])["edges"]
                EPlist.append(triplenum)
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
                triplenum = ast.literal_eval(data["afterSemanticMethodPreRetrievalModule"])["edges"]
                TPlist.append(triplenum)
                count += 1
            except Exception as e:
                count+=1
    TP = TP/count
    # print(NPlist)
    return get_median(NPlist),NP,get_median(EPlist),EP,get_median(TPlist),TP
    # return NP,EP,TP


def getexcel():
    for metric in metriclist:
        Result = [["NPTripleNum","NP","EPTripleNum","EP","TPTripleNum","TP"]]
        for windows in windowslist:
            pytypeResult = []
            # for prtype in prtypelist:
            NP =0
            EP =0
            TP=0
            NPtriple = 0
            EPtriple = 0
            TPtriple = 0
            for dataset in datasetlist:
                dicpath = f"{ROOTPATH}/{dataset}/subgraph/LLM_token_scale/qwen2-70b/EMB/Prune/{windows}.json"
                nptriplenum,np,eptriplenum,ep,tptriplenum,tp =singeInstance(dicpath,metric) #分别得到NP,EP,TP的值
                NP += np
                EP += ep
                TP += tp
                NPtriple += nptriplenum
                EPtriple += eptriplenum
                TPtriple += tptriplenum
            NP = NP/len(datasetlist)
            EP = EP/len(datasetlist)
            TP = TP/len(datasetlist)
            NPtriple = NPtriple/len(datasetlist)
            EPtriple = EPtriple/len(datasetlist)
            TPtriple = TPtriple/len(datasetlist)
            pytypeResult.append(nptriplenum)
            pytypeResult.append(NP)
            pytypeResult.append(eptriplenum)
            pytypeResult.append(EP)
            pytypeResult.append(tptriplenum)
            pytypeResult.append(TP)
            Result.append(pytypeResult)
        df = pd.DataFrame(Result[1:],columns=Result[0])
        df.to_excel(f"{ROOTPATH}/AverageSubgraph/Average-LLMModel-{metric}-16k.xlsx",index=False)
        print(f"{ROOTPATH}/AverageSubgraph/Average-LLMModel-{metric}-16k.xlsx")
def getImage():
    # 画一个一行两列的折线图，每一个折线图代表一个model的F1或者Recall
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1行6列，调整大小以适应
    axes = axes.flatten()  # 将 2D 数组转为 1D 数组，方便访问每个子图
    index = 0
    for metric in metriclist:
        excelpath = f"{ROOTPATH}/AverageSubgraph/Average-LLMModel-{metric}.xlsx"
        df = pd.read_excel(excelpath)
        x1 = df["NPTripleNum"].tolist()
        x2 = df["EPTripleNum"].tolist()
        x3 = df["TPTripleNum"].tolist()
        y1 = df["NP"].tolist()
        y2 = df["EP"].tolist()
        y3 = df["TP"].tolist()
        ax = axes[index]
        index += 1
        # if metric == "F1":
        #     ax.set_ylim(0.005, 0.25)
        #     ax.set_yticks(np.arange(0.005, 0.25, 0.05))
        # else:
        #     ax.set_ylim(0.3, 0.8)
        #     ax.set_yticks(np.arange(0.3, 0.81, 0.15))
        # ax.set_xlim(0, 17000)
        # ax.set_xticks(np.append(np.arange(0, 16000, 4000), 16000))

        # 设置不同的线型和标记
        ax.plot(x1, y1, label='NP', color='#2dabb2', marker='o', linestyle=':', linewidth=5, markersize=20)
        ax.plot(x2, y2, label='EP', color='#daab36', marker='s', linestyle='-', linewidth=5, markersize=20)
        ax.plot(x3, y3, label='TP', color='#f0552b', marker='*', linestyle='-.', linewidth=5, markersize=20)

        # 设置标题、标签和坐标轴的字号
        ax.set_xlabel("Subgraph Size (Number of the Tripes)", fontsize=20, fontweight='bold')
        ax.set_ylabel(f'{metric}', fontsize=40, fontweight='bold')
        ax.tick_params(axis='x', labelsize=30,length=10)
        ax.tick_params(axis='y', labelsize=30)

    # 统一添加图例
    fig.legend(["NP", "EP","TP"], loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=40)
    plt.subplots_adjust(wspace=0.25, hspace=0.55)
    plt.subplots_adjust(top=0.8, left=0.12, right=0.98,bottom=0.24)
    
    # plt.subplots_adjust(top=0.8,left=0.08,right=0.95)
    # plt.subplots_adjust(wspace=0.65, hspace=0.55)
    # plt.tight_layout()
    plt.savefig(f"{ROOTPATH}/AverageSubgraph/LLMModel.pdf")
    print(f"{ROOTPATH}/AverageSubgraph/LLMModel.pdf save!")
if __name__ == '__main__':
    ROOTPATH = "/back-up/gzy/dataset/VLDB/SE_new/"
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
    windowslist = [16000]
    metriclist = ["F1","Recall"]
    # 得到modellist 所有model的数据，每一个model存入两个excel中 分别为F1和Recall,excel格式为：
    # windows NP EP TP
    # 1000  0.1 0.2 0.3
    # 2000  0.2 0.3 0.4
    # ...
    getexcel()
    print("get excel done!")
    getImage()
    print("get image done!")