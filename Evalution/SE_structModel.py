import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    pprresult = 0
    pprtriple = []
    rwresult = 0
    rwtriple = []
    pprjsonpath = dicpath + "/PPR.json"
    rwjsonpath = dicpath + "/RandomWalk.json"
    # np
    with open(pprjsonpath,"r") as f:
        jsonf  = json.load(f)
        for data in jsonf:
            try:
                answer = data["query_info"]["answers"]
                subgraph = data["query_info"]["subgraph"]
                subgraphnode = getgraphnode(subgraph)
                triplenum = len(subgraph)
                pprtriple.append(triplenum)
                if metric == "F1":
                    pprresult += GetF1(answer,subgraphnode)
                elif metric == "Recall":
                    pprresult += GeRrecall(answer,subgraphnode)
                else:
                    print("metric error")
                count += 1
            except:
                pass
    # ep
    with open(rwjsonpath,"r") as f:
        jsonf  = json.load(f)
        for data in jsonf:
            try:
                answer = data["query_info"]["answers"]
                subgraph = data["query_info"]["subgraph"]
                subgraphnode = getgraphnode(subgraph)
                triplenum = len(subgraph)
                rwtriple.append(triplenum)
                if metric == "F1":
                    rwresult += GetF1(answer,subgraphnode)
                elif metric == "Recall":
                    rwresult += GeRrecall(answer,subgraphnode)
                else:
                    print("metric error")
            except:
                pass

    return pprresult/count,get_median(pprtriple), rwresult/count,get_median(rwtriple)


def getexcel():
    for metric in metriclist:
        Result = [["PPR Triple","PPR","RW Triple","RandomWalk"]]
        for windows in windowslist:
            PPRtriplenum = 0
            RWtriplenum = 0
            # pytypeResult = []
            # for prtype in prtypelist:
            PPR =0
            RandomWalk =0
            for dataset in datasetlist:
                dicpath = f"{ROOTPATH}/{dataset}/subgraph/SBE/{windows}/"
                pprreault,pprtriple,rwresult,rwtriple =singeInstance(dicpath,metric) #分别得到NP,EP,TP的值
                PPR += pprreault
                PPRtriplenum += pprtriple
                RandomWalk += rwresult
                RWtriplenum += rwtriple
            PPR = PPR/len(datasetlist)
            RandomWalk = RandomWalk/len(datasetlist)
            Result.append([PPRtriplenum,PPR,RWtriplenum,RandomWalk])
        df = pd.DataFrame(Result[1:],columns=Result[0])
        df.to_excel(f"{ROOTPATH}/AverageSubgraph/Average-StructModel-{metric}.xlsx",index=False)
        print(f"{ROOTPATH}/AverageSubgraph/Average-StructModel-{metric}.xlsx")
def getImage():
    # 画一个一行六列的折线图，每一个折线图代表一个model的F1或者Recall
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(30, 12))  # 1行6列，调整大小以适应
    axes = axes.flatten()  # 将 2D 数组转为 1D 数组，方便访问每个子图
    index = 0
    for metric in metriclist:
        excelpath = f"{ROOTPATH}/AverageSubgraph/Average-StructModel-{metric}.xlsx"
        df = pd.read_excel(excelpath)
        x1 = df["PPR Triple"].tolist()
        x2 = df["RW Triple"].tolist()
        y1 = df["PPR"].tolist()
        y2 = df["RandomWalk"].tolist()
        # y3 = df["TP"].tolist()
        ax = axes[index]
        index += 1
        if metric == "F1":
            ax.set_ylim(-0.005, 0.06)
        else:
            ax.set_ylim(0.18, 1.1)
        ax.set_xlim(0, 8000)
        # ax.set_xticks(np.append(np.arange(0, 6000, 2000), 9000))  # 在原有刻度的基础上加入 9000
        xticks = [1,1000,2000,3000, 4000,5000, 6000,7000, 8000, 8500]
        xticklabels = ['0','1000', '2000','3000', '4000','5000', '1.2e6', '2.7e7', '1.2e8','  ']
        ax.set_xticks(xticks, xticklabels)
        # xticks = list(xticks)
        # # xticks.append(90000)
        # ax.set_xticks(xticks) 
        # ax.set_xticks(np.arange(0, 6000, 2000))
        # 设置不同的线型和标记
        ax.plot(x1, y1, label='PPR', color='#9c4084', marker='o', linestyle=':', linewidth=15, markersize=30)
        ax.plot(x2, y2, label='RW', color='#3F81B4', marker='s', linestyle='-.', linewidth=15, markersize=30)
        # ax.plot(x, y3, label='TP', color='black', marker='*', linestyle=':', linewidth=5, markersize=15)
        ssize = 1000
        if metric == "Recall":
            ax.scatter(1983, 0.5536500513985316, label='KSE-1', color='#37a437', s=ssize, marker='s')
            ax.scatter(6000, 0.7605399907079977, label='KSE-2', color='#ff7f0d', s=ssize, marker='H')
            ax.scatter(7000, 0.8958490095949766, label='KSE-3', color='#6EBDB7', s=ssize, marker='P')
            ax.scatter(8000, 0.9831170954865532, label='KSE-4', color='#d62627', s=ssize, marker='D')
        elif metric == "F1":
            ax.scatter(1983,  0.044572105442986995, label='KSE-1', color='#37a437', s=ssize, marker='s')
            ax.scatter(6000, 0.006694404121798958, label='KSE-2', color='#ff7f0d', s=ssize, marker='H')
            ax.scatter(7000, 0.0037359566484027546, label='KSE-3', color='#6EBDB7', s=ssize, marker='P')
            ax.scatter(8000, 0.0031615108879198555, label='KSE-4', color='#d62627', s=ssize, marker='D')
        # 设置标题、标签和坐标轴的字号
        ax.set_xlabel('Subgraph Size (Number of the Triples)', fontsize=40, fontweight='bold')
        ax.set_ylabel(f'{metric}', fontsize=70, fontweight='bold')
        ax.tick_params(axis='x', labelsize=45,length=10,rotation=45)
        ax.tick_params(axis='y', labelsize=45,length=10)
        if metric == "F1":
            ax.set_yticks(np.arange(0, 0.07, 0.02))
        else:
            ax.set_yticks(np.arange(0, 1.1, 0.2))

    # 统一添加图例
    # 获取当前图例的句柄和标签
    handles, labels = plt.gca().get_legend_handles_labels()

    # 去除重复的图例
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    # 重新设置图例
    font_properties = FontProperties(weight='bold', size=50)
    plt.legend(unique_handles, unique_labels, loc='upper center', bbox_to_anchor=(-0.15, 1.34), ncol=6, prop=font_properties, labelspacing=0.01,columnspacing=0.5)
    plt.subplots_adjust(wspace=0.25, hspace=0.55)
    plt.subplots_adjust(top=0.8, left=0.1, right=0.95,bottom=0.25)
    plt.savefig(f"{ROOTPATH}/AverageSubgraph/StructModel.pdf")
    print(f"save {ROOTPATH}/AverageSubgraph/StructModel.pdf")
if __name__ == '__main__':
    ROOTPATH = "/back-up/gzy/dataset/VLDB/SE_new/"
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
    windowslist = [16,32,64,128,256]
    modellist = ["PPR","RandomWalk"]
    # prtypelist = ["node","edge","triple"]
    metriclist = ["F1","Recall"]
    # 得到modellist 所有model的数据，每一个model存入两个excel中 分别为F1和Recall,excel格式为：
    # windows PPR RandomWalk
    # 1  0.1 0.2 0.3
    # 2  0.2 0.3 0.4
    # ...
    # getexcel()
    # print("get excel done!")
    getImage()
    print("get image done!")