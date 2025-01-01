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
def processSingleJson(jsonpath,idlist):
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
            if id not in idlist:
                continue
            F1 += data["F1"]
            Hit += data["HR"]
            count += 1
    F1 = F1/count
    Hit = Hit/count
    return F1, Hit
def getidlist(dataset):
    txtPath = f"/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/Result/Instance_1000_{dataset}_sampleID.txt"
    idlist = []
    with open(txtPath, "r") as f:
        for line in f:
            idlist.append(line.strip())
    return idlist
def generation(PATHNUM,dataset):
    idlist = getidlist(dataset)
    for llm in llmlist:
        data = [["Method",f"F1",f"HR"]]
        for model in modelist:
            for prtype,result in resultlist.items():
                F1 = 0
                Hit32 = 0
                beforename = namedic[model]+"+"+prtype
                aftername = newnamedic[beforename]
            
                # for dataset in datasetlist:
                try:
                    # print("使用新数据：")
                    modelpath = f"/back-up/gzy/dataset/VLDB/Instance/NewGeneration/{dataset}/{model}"
                    jsonpath = f"{modelpath}/{resultlist[prtype]}/{llm}_{PATHNUM}_{shotsnum}_answers.json"
                    f1, hit32 = processSingleJson(jsonpath,idlist)
                    print("使用新数据成功！")
                except Exception as e:
                    # print(e)
                    # continue
                    print("新数据暂未推理,使用旧的数据：",aftername)
                    modelpath = f"/back-up/gzy/dataset/VLDB/Pipeline/Generation/{dataset}/{model}"
                    temp = resultlist[prtype].replace("EMB_v2","EMB")
                    jsonpath = f"{modelpath}/{temp}/{llm}_{PATHNUM}_{shotsnum}_answers.json"
                    f1, hit32 = processSingleJson(jsonpath,idlist)
                F1 += f1
                Hit32 += hit32
                # F1 = F1/len(datasetlist)
                # Hit32 = Hit32/len(datasetlist)

                data.append([aftername,F1,Hit32])
                # print(f"{prtype}处理完成！")
        # 按照aftername排序,排序规则参考newnamelist
        data = sorted(data,key=lambda x:newnamelist.index(x[0]))
        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_excel(f"/back-up/gzy/dataset/VLDB/Instance/NewGeneration/Result/{dataset}-{llm}-@{PATHNUM}.xlsx",index=False)
        # print(f"{dataset}-{llm} Excel处理完成！")
            
            
def GenerationImage(rootpath, saveImgpath,modeNUM):
    custom_colors = [
        "#293844",
        "#FAF0EB","#FCDCC6","#f79767","#f87349","#f64c29","#d32912","#DEBC33","#720909",
        "#42CFFE", "#0487e2",
        "#C8EAD1","#73C088","#397D54", "#3D4F2F"
    ]
    # 读取Excel数据
    # df = pd.read_excel(rootpath)
    # F1list = df[f"F1"].tolist()
    # Hitlist = df[f"HR"].tolist()
    # methodlist = df["Method"].tolist()
    # 一行两列的图,控制间隔
    fig, ax = plt.subplots(4, 4, figsize=(40, 28))


    for k in range(0,4):
        for i in range(0,4):
            if k ==0 and(i==0 or i==1 or i==2 or i==3):
                dataset = "CWQ"
            elif k ==1 and(i==0 or i==1 or i==2 or i==3):
                dataset = "webqsp"
            elif k ==2 and(i==0 or i==1 or i==2 or i==3):
                dataset = "GrailQA"
            elif k ==3 and(i==0 or i==1 or i==2 or i==3):
                dataset = "WebQuestion"
            llmpath = f"{rootpath}/{dataset}-{llmlist[i]}-@{PATHNUM}.xlsx"
            df = pd.read_excel(llmpath)
            Hitlist = df[f"HR"].tolist()
            x = np.arange(len(Hitlist), dtype=float)
            x[1:] += 1 
            x[9:] += 1 
            x[11:] += 1 
            x[15:] += 1 
            methodlist = df["Method"].tolist()
            # x = np.arange(len(Hitlist))
            # 设置坐标轴刻度，a为最小值，b为最大值
            a = min(Hitlist)-0.1
            b = max(Hitlist)+0.1
            if dataset == "CWQ":
                ax[k][i].set_ylim(0.32, 0.7)
                ax[k][i].set_yticks(np.arange(0.3, 0.75, 0.2))
            elif dataset == "webqsp":
                ax[k][i].set_ylim(0.32, 0.83)
                ax[k][i].set_yticks(np.arange(0.3, 0.75, 0.2))
            elif dataset == "GrailQA":
                ax[k][i].set_ylim(0.32, 0.75)
                ax[k][i].set_yticks(np.arange(0.3, 0.75, 0.2))
            elif dataset == "WebQuestion":
                ax[k][i].set_ylim(0.32, 0.75)
                ax[k][i].set_yticks(np.arange(0.3, 0.75, 0.2))
            ax[k][i].axvspan(-1, 0.7, facecolor='#B2B3B0', alpha=0.1)
            ax[k][i].axvspan(0.7, x[8] + 0.7, facecolor='#fee3ce', alpha=0.3)
            ax[k][i].axvspan(x[8] + 0.7, x[10] + 0.8, facecolor='#B7A368', alpha=0.1)
            ax[k][i].axvspan(x[10] + 0.8, x[-1]+1, facecolor='#E3E9D7', alpha=0.3)
            for j in range(len(Hitlist)):
                ax[k][i].bar(
                    x[j], Hitlist[j], width=0.8, 
                    color=custom_colors[j], edgecolor='black', 
                    linewidth=4, alpha=1
                )
            ax[k][i].set_xticks(x,)
            tempname = llmlist[i]
            tempname = tempname[0].upper() + tempname[1:]
            tempname = tempname.replace("Qwen2-70b","Qwen2-72B")
            tempname = tempname.replace("b","B")
            ax[k][i].set_xlabel(tempname, fontsize=45, fontweight='bold')
            if i == 0:
                ax[k][i].set_ylabel(f"Hits@1", fontsize=45, fontweight='bold')
            # 隐藏 X 轴刻度和标签
            ax[k][i].tick_params(axis='x', which='both', length=0)
            # ax[k][i].tick_params(axis='x', which='both', length=0)  # 隐藏刻度短线
            ax[k][i].set_xticklabels(range(1,16))  # 隐藏 X 轴标签
            ax[k][i].tick_params(axis='x', which='major', labelsize=30, width=3,length=20,rotation=60)
            ax[k][i].tick_params(axis='y', which='major', labelsize=40, width=3,length=20)
            ax[k][i].xaxis.labelpad = 10
            if dataset == "CWQ":
                ax[k][i].text(-0.5, 0.24, "No.", ha='right', va='center', fontsize=40, fontweight='bold')
            else:
                ax[k][i].text(-0.5, 0.21, "No.", ha='right', va='center', fontsize=40, fontweight='bold')
            # ax[k][i].text(-0.5, 0.15, "No.", ha='right', va='center', fontsize=40, fontweight='bold')

            ax[k][i].tick_params(axis='both', which='minor', labelsize=50, width=3)
            if  k == 0 and (i == 2):  # 在第一列和第二列之间插入
                # dataset_name = dataset_names[k * 2 + (i // 2)]  # 计算对应的数据集名称
                fig.text(
                    ax[k][i].get_position().x0 ,  # x 位置偏移值，适当调整
                    ax[k][i].get_position().y0-0.028  ,  # y 位置偏移值，适当调整
                    "(a) CWQ",
                    ha="center", va="center", 
                    fontsize=50, fontweight='bold'
                )
            elif  k == 1 and (i == 2):  # 在第一列和第二列之间插入
                # dataset_name = dataset_names[k * 2 + (i // 2)]  # 计算对应的数据集名称
                fig.text(
                    ax[k][i].get_position().x0 ,  # x 位置偏移值，适当调整
                    ax[k][i].get_position().y0-0.04 ,  # y 位置偏移值，适当调整
                    "(b) WebQSP",
                    ha="center", va="center", 
                    fontsize=50, fontweight='bold')
                
            elif  k == 2 and (i == 2):  # 在第一列和第二列之间插入
                # dataset_name = dataset_names[k * 2 + (i // 2)]  # 计算对应的数据集名称
                fig.text(
                    ax[k][i].get_position().x0 ,  # x 位置偏移值，适当调整
                    ax[k][i].get_position().y0-0.056 ,  # y 位置偏移值，适当调整
                    "(c) GrailQA",
                    ha="center", va="center", 
                    fontsize=50, fontweight='bold'
                )
            elif  k == 3 and (i == 2):  # 在第一列和第二列之间插入
                # dataset_name = dataset_names[k * 2 + (i // 2)]  # 计算对应的数据集名称
                fig.text(
                    ax[k][i].get_position().x0 ,  # x 位置偏移值，适当调整
                    ax[k][i].get_position().y0 -0.09 ,  # y 位置偏移值，适当调整
                    "(d) WebQuestions",
                    ha="center", va="center", 
                    fontsize=50, fontweight='bold'
                )

            # 画完之后清除画板
            # plt.sca(ax[k][i])
    y_positions = [0.5/64,15.5/64, 29.5/64, 43/64]  # 直接给定横线的y坐标

    # 在每个指定的y坐标位置画一条横线
    for y_position in y_positions:
        line_axis = fig.add_axes([0, y_position, 1, 0.001])  # 添加一个新的轴来绘制横线
        line_axis.plot([0, 10], [0, 0], color='black', lw=30)  # 设置线宽为 5
        line_axis.set_xticks([])  # 隐藏x轴刻度
        line_axis.set_yticks([])  # 隐藏y轴刻度
        line_axis.set_frame_on(False)  # 隐藏边框
    temp = []
    for i,method in enumerate(methodlist):
        temp.append(f"No.{i+1}:"+method)
    methodlist = temp
    # methodlist = ["No."+i for i in methodlist]
    legend_elements = [Patch(color=color, label=method) for color, method in zip(custom_colors, methodlist)]
    plt.subplots_adjust(wspace=0.3, hspace=0.75)  # 设置行和列之间的间隔
    font_properties = FontProperties(weight='bold', size=37)
    # plt.title(f"{dataset} Generation", fontsize=60, fontweight='bold')
    plt.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.54, 330), prop=font_properties,labelspacing=0.05)
    plt.subplots_adjust(top=0.87, left=0.08, right=0.99,bottom=0.1)
    plt.savefig(saveImgpath, format='pdf')
    # plt.tight_layout()
    print("画图保存成功：", saveImgpath)
if __name__ == "__main__":
    # llmlist = ["qwen2-7b","llama3-8b","qwen2-70b","qwen2-7b","llama3-8b","qwen2-70b"]
    llmlist = ["qwen2-7b","glm4-9b","qwen2-70b","llama3.3-70b","qwen2-7b","glm4-9b","qwen2-70b","llama3.3-70b"]
    # llmlist = ["llama3.3-70b"]
    datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
    # datasetlist = ["webqsp"]

    resultlist = {
        "GSR":"SPR",
        "OSR-EEMS":"SPR/EMB",
        "OSR-LLMS":"SPR/LLM/qwen2-70b/EMB_v2",
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
    # for PATHNUM in PATHNUMlist:
    #     for dataset in datasetlist:
    #         generation(PATHNUM,dataset)
    # print("全部处理完成！")
    # for llm in llmlist:
    # for dataset in datasetlist:
    GenerationImage(f"/back-up/gzy/dataset/VLDB/Instance/NewGeneration/Result",f"/back-up/gzy/dataset/VLDB/Instance/NewGeneration/Result/Generation-Hit@{PATHNUM}.pdf",len(llmlist))
    # print(f"{llm}画图完成！")