import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
showvalue = {
    7.3: "4",
    7.6: "5",
    8.2: "8",
}
def format_func(value, tick_number):
    if value<8:
        return r'$2^{%d}$' % int(showvalue[value]) 
    return r'$2^{%d}$' % int(value)
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
    npresult = 0
    epresult = 0
    tpresult = 0
    npjsonpath = dicpath + "/node.json"
    epjsonpath = dicpath + "/edge.json"
    tpjsonpath = dicpath + "/triple.json"
    NPlist = []
    EPlist = []
    TPlist = []
    # np
    with open(npjsonpath,"r") as f:
        jsonf  = json.load(f)
        for data in jsonf:
            try:
                answer = data["query_info"]["answers"]
                subgraph = data["query_info"]["subgraph"]
                subgraphnode = getgraphnode(subgraph)
                if metric == "F1":
                    npresult += GetF1(answer,subgraphnode)
                elif metric == "Recall":
                    npresult += GeRrecall(answer,subgraphnode)
                else:
                    print("metric error")
                triplenum = len(subgraph)
                NPlist.append(triplenum)
                count += 1
            except Exception as e:
                print(e)
                count+=1
    # ep
    npresult = npresult/count
    count = 0
    with open(epjsonpath,"r") as f:
        jsonf  = json.load(f)
        for data in jsonf:
            try:
                answer = data["query_info"]["answers"]
                subgraph = data["query_info"]["subgraph"]
                subgraphnode = getgraphnode(subgraph)
                if metric == "F1":
                    epresult += GetF1(answer,subgraphnode)
                elif metric == "Recall":
                    epresult += GeRrecall(answer,subgraphnode)
                else:
                    print("metric error")
                triplenum = len(subgraph)
                EPlist.append(triplenum)
                count += 1
            except:
                # print
                count+=1
    epresult = epresult/count
    count = 0
    # tp
    with open(tpjsonpath,"r") as f:
        jsonf  = json.load(f)
        for data in jsonf:
            try:
                answer = data["query_info"]["answers"]
                subgraph = data["query_info"]["subgraph"]
                subgraphnode = getgraphnode(subgraph)
                if metric == "F1":
                    tpresult += GetF1(answer,subgraphnode)
                elif metric == "Recall":
                    tpresult += GeRrecall(answer,subgraphnode)
                else:
                    print("metric error")
                triplenum = len(subgraph)
                TPlist.append(triplenum)
                count += 1
            except:
                count+=1
    tpresult = tpresult/count
    # print("NPresult",NPlist)
    return get_median(NPlist),npresult,get_median(EPlist),epresult,get_median(TPlist),tpresult
def getppr(dicpath,metric):
    # 计算F1,recall,和coverrate
    count = 0
    pprresult = 0
    pprtriple = []
    pprjsonpath = dicpath + "/ppr.json"
    with open(pprjsonpath,"r") as f:
        jsonf  = json.load(f)
        for data in jsonf:
            try:
                answer = data["query_info"]["answers"]
                subgraph = data["query_info"]["subgraph"]
                subgraphnode = getgraphnode(subgraph)
                if metric == "F1":
                    pprresult += GetF1(answer,subgraphnode)
                elif metric == "Recall":
                    pprresult += GeRrecall(answer,subgraphnode)
                else:
                    print("metric error")
                triplenum = len(subgraph)
                pprtriple.append(triplenum)
                count += 1
            except:
                count+=1
    return get_median(pprtriple),pprresult/count
def getpprexcel():
    modellist = ["EMB"]
    for metric in metriclist:
        for model in modellist:
            Result = [["PPRTriple","PPR"]]
            for windows in windowslist:
                pytypeResult = []
                # for prtype in prtypelist:
                PPR =0
                PPRTriple = 0
                for dataset in datasetlist:
                    dicpath = f"{ROOTPATH}/{dataset}/subgraph/EEMs/{windows}/{model}/"
                    pprtriple,ppr =getppr(dicpath,metric) #分别得到NP,EP,TP的值
                    PPR += ppr
                    PPRTriple += pprtriple
                PPR = PPR/len(datasetlist)
                PPRTriple = PPRTriple/len(datasetlist)
                pytypeResult.append(PPRTriple)
                pytypeResult.append(PPR)
                Result.append(pytypeResult)
            df = pd.DataFrame(Result[1:],columns=Result[0])
            df.to_excel(f"{ROOTPATH}/AverageSubgraph/Average-smallModel-PPR-{metric}.xlsx",index=False)
            print(f"{ROOTPATH}/AverageSubgraph/Average-smallModel-PPR-{metric}.xlsx")
def getexcel():
    for metric in metriclist:
        for model in modellist:
            Result = [["NPTripleNum","NP","EPTripleNum","EP","TPTripleNum","TP"]]
            for windows in windowslist:
                pytypeResult = []
                # for prtype in prtypelist:
                NPA =0
                EPA =0
                TPA =0
                Nptriple =0
                Eptriple =0
                Tptriple =0
                for dataset in datasetlist:
                    dicpath = f"{ROOTPATH}/{dataset}/subgraph/EEMs/{windows}/{model}/"
                    nptriple,npresult,eptriple,epresult,tptriple,tpresult =singeInstance(dicpath,metric) #分别得到NP,EP,TP的值
                    NPA += npresult
                    EPA += epresult
                    TPA += tpresult
                    Nptriple += nptriple
                    Eptriple += eptriple
                    Tptriple += tptriple
                NPA = NPA/len(datasetlist)
                EPA = EPA/len(datasetlist)
                TPA = TPA/len(datasetlist)
                Nptriple = Nptriple/len(datasetlist)
                Eptriple = Eptriple/len(datasetlist)
                Tptriple = Tptriple/len(datasetlist)
                pytypeResult.append(Nptriple)
                pytypeResult.append(NPA)
                pytypeResult.append(Eptriple)
                pytypeResult.append(EPA)
                pytypeResult.append(Tptriple)
                pytypeResult.append(TPA)
                Result.append(pytypeResult)
            df = pd.DataFrame(Result[1:],columns=Result[0])
            df.to_excel(f"{ROOTPATH}/AverageSubgraph/Average-{model}-{metric}.xlsx",index=False)
            print(f"{ROOTPATH}/AverageSubgraph/Average-{model}-{metric}.xlsx")
def getImage():
    # 画一个一行六列的折线图，每一个折线图代表一个model的F1或者Recall
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(42, 18))  # 1行6列，调整大小以适应
    axes = axes.flatten()  # 将 2D 数组转为 1D 数组，方便访问每个子图
    index = 0
    modelname = {
        "EMB":"ST",
        "BGE":"BGE",
        "BM25":"BM25"
    }
    markers = [
    'o',  # 圆圈
    '^',  # 上三角
    'v',  # 下三角
    's',  # 方形
    'P',  # 五边形
    '*',  # 星号
    '+',  # 加号
    'x',  # 交叉
    'D',  # 菱形
    'H',  # 六边形
    'h',  # 小六边形
    '|',  # 垂直线
    '_',  ]
    linestyles = [
    '-',   # 实线
    '--',  # 虚线
    '-.',  # 点划线
    ':']
    colorlist = ["#4169E1","#FF8C00","#32CD32","#FF0000","#0000FF","#FFD700","#FF69B4","#8A2BE2","#00FF00","#FF4500","#FF6347","#FF00FF"]
    color1 = "#2dabb2"
    color2 = "#daab36"
    color3 = "#f0552b"
    markline = [
        ['o','-',color1],
        ['^','--',color1],
        ['v','-.',color1],
        ['P','-',color2],
        ['*','--',color2],
        ['+','-.',color2],
        ['x','-',color3],
        ['H','--',color3],
        ['h','-.',color3],
        ['D',':',"#181C1A"]
    ]
    all_handles = []
    all_labels = []
    for metric in metriclist:
        i=0
        ax = axes[index]
        # color1
        if metric == "F1":
            pprf1excel = "/back-up/gzy/dataset/VLDB/SE_new/AverageSubgraph/Average-smallModel-PPR-F1.xlsx"
            dfpprf1 = pd.read_excel(pprf1excel)
            # y0 = dfpprf1["PPR"].tolist()
            # x0 = dfpprf1["PPRTriple"].tolist()
            # x0 = np.log2(x0)
            #line, = ax.plot(x0, y0, label=f'PPR', color=markline[9][2], marker=markline[9][0], linestyle=markline[9][1], linewidth=5, markersize=20)
            # all_handles.append(line)
            # all_labels.append('PPR')
            # 画三个散点
            ssize = 2000
            ax.scatter(7.3, 0.05, label='LLM-TP', color='#FF00FF', s=ssize, marker='P')
            ax.scatter(7.6, 0.045, label='LLM-NP', color='#FF4500', s=ssize, marker='s')
            ax.scatter(8.0, 0.04, label='LLM-EP', color='#8A2BE2', s=ssize, marker='H')
            all_handles.extend(ax.collections[-3:])  # Add scatter plot handles
            all_labels.extend(['LLM-TP', 'LLM-NP', 'LLM-EP'])
            yticks = [0.004,0.01,0.02,0.03,0.04,0.045,0.05]
            yticklabels = ["0","0.01","0.02","0.03","0.06","0.18","0.20"]
            ax.set_yticks(yticks, yticklabels)
        elif metric == "Recall":
            pprrecall = "/back-up/gzy/dataset/VLDB/SE_new/AverageSubgraph/Average-smallModel-PPR-Recall.xlsx"
            dfpprrecall = pd.read_excel(pprrecall)
            # y0 = dfpprrecall["PPR"].tolist()
            # x0 = dfpprrecall["PPRTriple"].tolist()
            # x0 = np.log2(x0)
            #line ,=ax.plot(x0, y0, label=f'PPR', color=markline[9][2], marker=markline[9][0], linestyle=markline[9][1], linewidth=10, markersize=40)
            # all_handles.append(line)
            # all_labels.append('PPR')
            ssize = 2000
            # ax.scatter(np.log2(21), 0.49, label='LLM-NP', color='#FF4500', s=ssize, marker='s')
            # ax.scatter(np.log2(289), 0.70, label='LLM-EP', color='#8A2BE2', s=ssize, marker='H')
            # ax.scatter(np.log2(14), 0.50, label='LLM-TP', color='#FF00FF', s=ssize, marker='P')
            ax.scatter(7.3, 0.50, label='LLM-TP', color='#FF00FF', s=ssize, marker='P')
            ax.scatter(7.6, 0.49, label='LLM-NP', color='#FF4500', s=ssize, marker='s')
            ax.scatter(8.0, 0.70, label='LLM-EP', color='#8A2BE2', s=ssize, marker='H')
            all_handles.extend(ax.collections[-3:])  # Add scatter plot handles
            all_labels.extend(['LLM-TP', 'LLM-NP', 'LLM-EP'])
            # yticks = [i for i in np.range(0.4,0.85,0.2)]
            # yticklabels = [str(i) for i in yticks]
            # ax.set_yticks(yticks, yticklabels)
            ax.set_yticks(np.arange(0.4, 0.81, 0.1))
        for model in modellist:
            excelpath = f"{ROOTPATH}/AverageSubgraph/Average-{model}-{metric}.xlsx"
            df = pd.read_excel(excelpath)
            # x = df["windows"].tolist()
            x1 = df["NPTripleNum"].tolist()
            x2 = df["EPTripleNum"].tolist()
            x3 = df["TPTripleNum"].tolist()
            x1 = np.log2(x1)
            x2 = np.log2(x2)
            x3 = np.log2(x3)
            y1 = df["NP"].tolist()
            y2 = df["EP"].tolist()
            y3 = df["TP"].tolist()
            
            if metric == "F1":
                ax.set_ylim(0.004, 0.054)
            else:
                ax.set_ylim(0.4, 0.81)
            # 设置不同的线型和标记
            line1,=ax.plot(x1, y1, label=f'{modelname[model]}-NP', color=markline[i][2], marker=markline[i][0], linestyle=markline[i][1], linewidth=10, markersize=40)
            i+=1
            # print(i)
            line2,=ax.plot(x2, y2, label=f'{modelname[model]}-EP', color=markline[i][2], marker=markline[i][0], linestyle=markline[i][1], linewidth=10, markersize=40)
            i+=1
            line3,=ax.plot(x3, y3, label=f'{modelname[model]}-TP', color=markline[i][2], marker=markline[i][0], linestyle=markline[i][1], linewidth=10, markersize=40)
            i+=1
            all_handles.extend([line1, line2, line3])
            all_labels.extend([f'{modelname[model]}-NP', f'{modelname[model]}-EP', f'{modelname[model]}-TP'])
            # 设置标题、标签和坐标轴的字号
            ax.set_xlabel("Subgraph Size (Number of the Tripes)", fontsize=65, fontweight='bold')
            ax.set_ylabel(f'{metric}', fontsize=90, fontweight='bold')
            ax.tick_params(axis='y', labelsize=70,length=10)
            ax.tick_params(axis='x', labelsize=70,length=10,rotation=45)
            # ax.set_xlim(0, 5000)
            # ax.set_xticks(np.append(np.arange(0, 6000, 2000), 9000))  # 在原有刻度的基础上加入 9000
            # xticks = [1,1000,2000,3000, 4000,5000, 6000,7000, 8000, 8500]
            # xticklabels = ['0','1000', '2000','3000', '4000','5000', '1.2e6', '2.7e7', '1.2e8','  ']
            # ax.set_xticks(xticks, xticklabels)
            # xticks = windowslist
            xticks = [7.3,7.6,8,9,10,11,12]
            xticklabels = ["14","21","8","9","10","11","12"]
            ax.set_xticks(xticks, xticklabels)
            ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        index += 1
    # 统一添加图例
    #legendlist = ["PPR","LLM-TP","LLM-NP","LLM-EP","ST-NP", "ST-EP", "ST-TP", "Re-Ranker-NP", "Re-Ranker-EP", "Re-Ranker-TP", "BM25-NP", "BM25-EP", "BM25-TP"]
    # all_handles_after = [all_handles[2],,all_handles[7],all_handles[10],
    #                      all_handles[1],all_handles[6],all_handles[9],all_handles[12],
    #                      all_handles[3],all_handles[5],all_handles[8],all_handles[11],
    #                     all_handles[0]]
    # print(all_handles)
    print(all_labels)
    handles, labels = ax.get_legend_handles_labels()
    handles_after = [all_handles[2],all_handles[3],all_handles[1],
                     all_handles[4],all_handles[5],all_handles[6],
                     all_handles[7],all_handles[8],all_handles[9],
                    all_handles[10],all_handles[11],all_handles[12]]
    labels_after = [all_labels[2],all_labels[3],all_labels[1],
                    all_labels[4],all_labels[5],all_labels[6],
                    all_labels[7],all_labels[8],all_labels[9],
                    all_labels[10],all_labels[11],all_labels[12]]
    # all_handles_after = [all_handles[2],all_handles[3],all_labels[1],all_labels[0],
    #                      all_handles[4],all_handles[5],all_labels[6],
    #                      all_handles[7],all_handles[8],all_labels[9],
    #                     all_handles[10],all_handles[11],all_labels[12]]
    # all_labels_after = [all_labels[2],all_labels[3],all_labels[1],all_labels[0],
    #                     all_labels[4],all_labels[5],all_labels[6],
    #                     all_labels[7],all_labels[8],all_labels[9],
    #                     all_labels[10],all_labels[11],all_labels[12]]
    # print(all_labels_after)
    # print(len(set(all_handles_after)))
    font_properties = FontProperties(weight='bold', size=65)
    fig.legend(handles=handles_after, labels=labels_after,loc='upper center', bbox_to_anchor=(0.52, 0.95), ncol=4, prop=font_properties)
    # plt.subplots_adjust(top=0.8, left=0.08, right=0.95,bottom=0.14)
    plt.subplots_adjust(top=0.65, left=0.1, right=0.95,bottom=0.22)
    plt.subplots_adjust(wspace=0.25, hspace=0.55)
    # plt.tight_layout()
    plt.savefig(f"{ROOTPATH}/AverageSubgraph/SmallModel.pdf")
    print(f"save {ROOTPATH}/AverageSubgraph/SmallModel.pdf")
if __name__ == '__main__':
    ROOTPATH = "/back-up/gzy/dataset/VLDB/SE_new/"
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
    windowslist = [100,200,400,800]
    modellist = ["EMB","BGE","BM25"]
    prtypelist = ["node","edge","triple"]
    metriclist = ["F1","Recall"]
    # 得到modellist 所有model的数据，每一个model存入两个excel中 分别为F1和Recall,excel格式为：
    # windows NP EP TP
    # 500   0.1 0.2 0.3
    # 1000  0.2 0.3 0.4
    # ...
    # getexcel()
    # print("get excel done!")
    # getpprexcel()
    # print("get ppr excel done!")
    getImage()
    print("get image done!")