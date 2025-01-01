# /home/lzy/expResult/NewResult/CWQ/PPR2/Dij/Emb_FT/paths-PPR2-Dij-Emb_FT.json
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from evalut import eval_hr_topk
from evalut import eval_f1
from matplotlib.font_manager import FontProperties
def processSingeJson(jsonpath,idlist):
    with open(jsonpath, 'r') as file:
            data = json.load(file)
    count = 0
    PostRetrievalModuleF1 = 0
    for i in range(len(data["eval_info"])):
        id = str(data["eval_info"][i]["id"])
        # print(id)   
        if id not in idlist:
            continue
        else:
            # print("id:",id)
            PostRetrievalModuleF1 += data["eval_info"][i]["PostRetrievalModuleF1"]
            count +=1
    print(jsonpath.split("/")[5]," ",jsonpath.split("/")[-1],"count:",count)
    return PostRetrievalModuleF1/count
def getidlist(instancenum,dataset):
    txtPath = f"/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/Result/{instancenum}_{dataset}_sampleID.txt"
    idlist = []
    with open(txtPath, "r") as f:
        for line in f:
            idlist.append(line.strip())
    return idlist
def getSingleAgentJson(jsonpath,idlist):
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
            # if prtype=="GSR":
            #     # 随机打乱
            #     # print("random shuffle")
            #     # print("len(prediction)",len(prediction))
            #     random.shuffle(prediction)
            prediction = prediction[:PATHNUM]
            F1 += eval_f1(prediction,answers)
            Hit32 += eval_hr_topk(prediction,answers,PATHNUM)
            count += 1
    F1 = F1/count
    Hit32 = Hit32/count
    # print(dataset,jsonpath,count)
    return F1, Hit32
def getAgent(agentpath,idlist):
    # modelist = ["EMB/edge","PPR","LLM/qwen2-70b/EMB/ppr_1000_edge_64"]
    modelist = ["PPR"]
    F1= 0
    for model in modelist:
        modelpath = f"{agentpath}/{model}"
        jsonpath = f"{modelpath}/SPR/LLM/qwen2-70b/Agent16_v2.json"
        try:
            f1, hit32 = getSingleAgentJson(jsonpath,idlist= idlist)
        except Exception as e:
            print(e)
            continue
        F1 += f1
    F1 = F1/len(modelist)
    # Hit32 = Hit32/len(modelist)
    return F1
def getexcel(instancenum):
    for dataset in datasetlist:
        idlist = getidlist(instancenum,dataset)
        result = [["Method","F1"]]
        for jsonname in jsonlist:
            if "paths" in jsonname:
                prename = jsonname.split("-")[1]
                rename = jsonname.split("-")[2]
                postname = jsonname.split("-")[3].split(".")[0]
                jsonpath = f"{ROOTPATH}/{dataset}/{prename}/{rename}/{postname}/{jsonname}"
                f1 = processSingeJson(jsonpath,idlist)
                # result.append([jsonname,f1])
            else:
                agentpath = "/back-up/gzy/dataset/VLDB/new250/PathRetrieval/webqsp"
                f1 = getAgent(agentpath,idlist)
            result.append([jsonname,f1])
        df = pd.DataFrame(result[1:],columns=result[0])
        instancename = instancenum.split("_")[1]
        df.to_excel(f"{SAVEPATH}/{dataset}-Agent-{instancename}.xlsx",index=False)
        print(f"{SAVEPATH}/{dataset}-Agent-{instancename}.xlsx save!")
def getImage(instancenum):
    datasetlist = ["webqsp"]
    for dataset in datasetlist:
        instancename = instancenum.split("_")[1]
        excelpath = f"{SAVEPATH}/{dataset}-Agent-{instancename}.xlsx"
        # 画柱状图
        df = pd.read_excel(excelpath)
        methoddic = {
            "SPR/LLM/qwen2-70b/Agent16_v2.json":"Agent",
            "paths-PPR2-Dij-LLM.json":"LLM",
            "paths-PPR2-Dij-LLM_FT.json":"LLM-FT"
        }
        methodlist = df["Method"]
        methodlist = [methoddic[method] for method in methodlist]
        F1list = df["F1"]
        namedic = {
            "LLM":"LLM(one-call)",
            "LLM-FT":"LLM-FT(one-call)",
            "Agent":"LLM(multiple)"
        }
        # 画图
        plt.figure(figsize=(10, 5))
        # for循环画柱子，每个柱子不同颜色，并根据颜色生成图例
        colorlist = ["#9ED17B","#3D9F3C","#9DC7DD","#367DB0","#F9C784","#F9A784"]
        for i in range(len(methodlist)):
            plt.bar(methodlist[i], F1list[i], color=colorlist[i],label=namedic[methodlist[i]],width=0.5,edgecolor='black', 
                    linewidth=4, alpha=1,)
        # plt.bar(methodlist, F1list, color='steelblue')
        # plt.xlabel('Method')
        # plt.xlabel('Method',fontsize=30,fontweight='bold')
        plt.ylabel('F1',fontsize=40,fontweight='bold')
        # plt.title(f'{dataset} F1 of different methods')
        plt.ylim(0.2, 0.5)
        plt.yticks(fontsize=40)
        # ticks = 
        labels = ["LLM\n(one-call)", "LLM-FT\n(one-call)", "LLM\n(multiple)"]
        plt.xticks(methodlist,labels, fontsize=30,fontweight='bold')
        # for tick, label in zip(methodlist, labels):
        #     label_parts = label.split("\n")  # 按照换行符分割标签
        #     # 创建格式化的标签：第一行字体大，第二行字体小
        #     formatted_label = f'${label_parts[0]}$' + '\n' + f'${label_parts[1]}$'
        #     plt.text(tick, -0.02, formatted_label, ha='center', va='top', fontsize=30, fontweight='bold')
        # 添加图例
        font_properties = FontProperties(weight='bold', size=20) #bbox_to_anchor=(0.5, 1.03)
        # plt.legend(loc='upper left',prop=font_properties,bbox_to_anchor=(0, 1.65),ncol=1,labelspacing=0.05)
        plt.subplots_adjust(top=0.95, left=0.2, right=0.94,bottom=0.24)
        # plt.tight_layout()
        plt.savefig(f"{SAVEPATH}/{dataset}-Agent-{instancename}.pdf",format="pdf")
        print(f"{SAVEPATH}/{dataset}-Agent-{instancename}.pdf save!")
        
    # excelpath = f"{SAVEPATH}/WebQuestion-Agent-{instancenum.split('_')[1]}.xlsx"

if __name__ == "__main__":
    datasetlist = ["webqsp"]
    INstancelist = ["PR_250"]
    jsonlist = [
        "paths-PPR2-Dij-LLM.json",
        "paths-PPR2-Dij-LLM_FT.json",
        "SPR/LLM/qwen2-70b/Agent16_v2.json"
    ]
    PATHNUM = 16
    ROOTPATH = "/home/lzy/expResult/NewResult"
    SAVEPATH = "/back-up/gzy/dataset/VLDB/new250/PathRetrieval/Result"
    # for instancenum in INstancelist:
    #     getexcel(instancenum)
    for instancenum in INstancelist:
        getImage(instancenum)
