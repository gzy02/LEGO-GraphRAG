import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
# todo: 写一个函数，输入的是excel文件地址，输出的是折叠柱状图，每个dataset画一个柱状图，因此需要画一个一行四列的柱状图
# excel 文件的格式是：第一列是方法的名称，第二列和第三个列根据具体的任务去确定：
# Time: SETime,PRTime
# Tokens : EEMs,LLMs
# Mermoery: All,Average
def format_func(value, tick_number):
    return r'$2^{%d}$' % int(value)
def plotImage(tasktype, excelRoot, savename):
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
    # 画堆积柱状图，一个有四个dataset,因此需要画一个一行四列的柱状图
    fig, axes = plt.subplots(1, 4, figsize=(30,9))  # 1行4列，调整大小以适应
    axes = axes.flatten()  # 将 2D 数组转为 1D 数组，方便访问每个子图
    
    # 数据集列表，需要替换为你实际使用的数据集
    # datasetlist = ['reason_dataset1', 'reason_dataset2', 'reason_dataset3', 'reason_dataset4']
    numdic= {
        0:"(a)",
        1:"(b)",
        2:"(c)",
        3:"(d)"
    }
    datasetdic = {
        "CWQ":"CWQ",
        "webqsp":"WebQSP",
        "GrailQA":"GrailQA",
        "WebQuestion":"WebQuestions"
    }
    # 开始画每一个子图
    for i, dataset in enumerate(datasetlist):
        excelpath = f"{excelRoot}/{dataset}-{tasktype}.xlsx"
        df = pd.read_excel(excelpath.replace("reason_dataset", dataset))
        # namelist = df["Instance"].tolist()
        ax = axes[i]
        if tasktype == "Time":
            Value1 = df["SETime"].tolist()  # SETime
            Value1 = [x+75.29 for x in Value1]
            Value2 = df["PRTime"].tolist()  # PRTime
            label1 = "Subgraph-Extraction Time"
            label2 = "Path-Retrieval Time"
            hatch1 = "\\\\"
            hatch2 = "  "
            ax.set_ylim(40, 240)
            maker1 = "◆"
            maker1color = "purple"
            maker2 = "★"
            maker2color = "#A52A2A"
            ylabelname = "Time (s)"
            ax.set_yticks(np.arange(40, 245, 50))
        elif tasktype == "Token":
            Value1 = df["EEMs Token"].tolist()  # EEMs
            Value1 = [math.log2(int(x)) if int(x) > 0 else 0 for x in Value1]

            Value2 = df["LLMs Token"].tolist()  # LLMs
            Value2 = [math.log2(int(x)) if int(x) > 0 else 0 for x in Value2]
            label1 = "End-to-End Models(EEMs)"
            label2 = "Large Language Models(LLMs)"
            hatch1 = "//"
            hatch2 = "  "
            ax.set_ylim(0, 23)
            maker1 = "▼"
            # maker2 = "★"
            maker1color = "#1118EE"
            maker2color = "#A52A2A"
            ylabelname = r"Token"
            ax.set_yticks(np.arange(0, 24, 5))

            
        elif tasktype == "Memory":
            Value1 = df["PeakMemory"].tolist()  # All
            Value1 = [int(x)/1073741824 for x in Value1]
            # print(Value1)
            Value2 = df["AverageMemory"].tolist()  # Average
            Value2 = [int(x)/1073741824 for x in Value2]
            # print()
            label1 = "Peak"
            label2 = "Average"
            hatch1 = "////"
            hatch2 = "  "
            ax.set_ylim(0, 100)
        else:
            raise ValueError("tasktype must be one of Time, Tokens, Memory")
        # Value1 = df["SETime"].tolist()  # SETime
        # Value2 = df["PRTime"].tolist()  # PRTime
        
        x = np.arange(len(Value1), dtype=float)
        x[1:] += 1 
        x[9:] += 1 
        x[11:] += 1 
        x[15:] += 1 
        # print(x[-1])
        ax.axvspan(-1, 1, facecolor='#B2B3B0', alpha=0.1)
        ax.axvspan(1, 20, facecolor='#fee3ce', alpha=0.3)
        ax.axvspan(20, 26, facecolor='#B7A368', alpha=0.1)
        ax.axvspan(26 , 37, facecolor='#E3E9D7', alpha=0.3)
        # x = np.arange(len(Hitlist))
        # 设置坐标轴刻度，a为最小值，b为最大值
        a = min(Value1)-0.1
        b = max(Value2)+0.1
        # print("Value1:",Value1)
        # print("Value2:",Value2)
        # Value1 = [x for pair in zip(Value1, Value2) for x in pair]
        # print(Value1)
        width = 1.6  # 控制柱子的宽度
        gap = 0   # 设置柱子之间的间隔
        # print(x)
        x = [i*2 for i in x]
        for j in range(len(Value1)):
            if tasktype != "Memory":
                # 第一组柱子
                ax.bar(
                    x[j], Value1[j], width=width, 
                    color=custom_colors[j], edgecolor='black', 
                    linewidth=2, alpha=1, label=label1, hatch=hatch1
                )
                ax.bar(
                    x[j], Value2[j],bottom=Value1[j], width=width, 
                    color=custom_colors[j], edgecolor='black', 
                    linewidth=2, alpha=1, label=label2, hatch=hatch2
                )
        ax.set_xticks(x,)
        ax.set_xticklabels(range(1,16))
        if i == 0:
            ax.set_ylabel(ylabelname, fontsize=60, fontweight='bold')
        if tasktype == "Token":
            ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        ax.tick_params(axis='x', which='major', labelsize=30, width=3,length=10,rotation=70)
        ax.tick_params(axis='y', which='major', labelsize=40, width=3,length=0)
        if tasktype == "Time":
            ax.text(16, -10, numdic[i]+" "+datasetdic[dataset], ha='center', va='center', fontsize=50, fontweight='bold')
            ax.text(-0.9,25, "No.", ha='right', va='center', fontsize=25, fontweight='bold')
            
        elif tasktype == "Token":
            ax.text(16, -5.5, numdic[i]+" "+datasetdic[dataset], ha='center', va='center', fontsize=50, fontweight='bold')
            ax.text(-0.8,-2, "No.", ha='right', va='center', fontsize=25, fontweight='bold')
    if tasktype == "Time":
        handles = [
        Patch(hatch=hatch1, label=label1, facecolor='none',edgecolor='black'),  # 空白填充
        Patch(hatch=hatch2, label=label2, facecolor='none',edgecolor='black'),   # 空白填充
        Patch(hatch=maker1, label="               ", facecolor='none')]
        # num1 = -25
        # num2 = 169
        # ax.text(num1,num2, maker1, color=maker1color, fontsize=30, ha='center'),
        # ax.text(num1+7,num2-0.5, ":<0.1s", color='black', fontsize=30, ha='center',fontweight='bold')
        # ax.text(num1+16,num2, maker2, color=maker2color, fontsize=30, ha='center'),
        # ax.text(num1+22,num2-0.5, ":<1s", color='black', fontsize=30, ha='center',fontweight='bold')
        
    elif tasktype == "Token":
        handles = [
        Patch(hatch=hatch1, label=label1, facecolor='none',edgecolor='black'),  # 空白填充
        Patch(hatch=hatch2, label=label2, facecolor='none',edgecolor='black'),   # 空白填充
        Patch(hatch=maker1, label="               ", facecolor='none')]
        # num1 = -27
        # num2 = 16.8
        # ax.text(num1,num2, maker1, color=maker1color, fontsize=30, ha='center'),
        # ax.text(num1+8,num2, ":0 token", color='black', fontsize=30, ha='center',fontweight='bold')
    handles = handles[:3]
    # plt.legend(by_label.values(), by_label.keys())
    font_properties = FontProperties(weight='bold', size=50)
    
    plt.subplots_adjust(wspace=0.2, hspace=0.55,)
    plt.legend(handles=handles, loc='upper center',bbox_to_anchor=(-0.95, 1.25), ncol=3, prop=font_properties,labelspacing=0,frameon=False)
    plt.subplots_adjust(top=0.85, left=0.08, right=0.98,bottom=0.2)
    plt.tight_layout()
    plt.savefig(savename, format='pdf')
    print("save",savename)
if __name__ == "__main__":
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
    excelRoot= f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp"   
    tasklist = ["Time","Token"]
    for tasktype in tasklist:
        print(f"==========开始处理{tasktype}==========")
        savename = f"{excelRoot}/{tasktype}_INone.pdf"
        plotImage(tasktype,excelRoot,savename)
        print(f"=========={tasktype}处理完成！==========")
    print("全部处理完成！")

