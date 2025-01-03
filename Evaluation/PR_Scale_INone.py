# 把excel不同scale合并到一个excel
# 然后画折线图
# 只画F1,GSR, OSR-EEMS, ISR-EEMS
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 把不同scale的excel合并到一个excel
import pandas as pd
import re
def merge_excel_files_by_method(file_paths, output_file):
    """
    根据 'Method' 列合并多个 Excel 文件，并将新的指标列添加到已有列之后。
    
    参数：
        file_paths (list of str): 需要合并的 Excel 文件路径列表。
        output_file (str): 保存合并结果的输出文件路径。
    """
    # 读取第一个文件，用于初始化合并后的 DataFrame
    first_df = pd.read_excel(file_paths[0])
    filtered_columns = ['Method'] + [col for col in first_df.columns if 'F1' in col]
    merged_df = first_df[filtered_columns]
    
    # 遍历剩余文件，逐个与已有 DataFrame 合并
    for file in file_paths[1:]:
        # 读取当前 Excel 文件
        current_df = pd.read_excel(file)
        filtered_columns = ['Method'] + [col for col in current_df.columns if 'F1' in col]
        current_df = current_df[filtered_columns]
        # 使用 'Method' 列进行合并，保留所有数据（外连接合并）
        merged_df = pd.merge(merged_df, current_df, on='Method', how='outer')
    # 将最终合并的 DataFrame 保存为 Excel 文件
    merged_df.to_excel(output_file, index=False)
    print(f"合并后的 Excel 文件已保存到: {output_file}")
def mergeExcel():
    for method in methodlist:
        filelist = []
        for pathnum in PATHNUM:
            filepath = f"{rootpath}/Average-{method}-@{pathnum}.xlsx"
            filelist.append(filepath)
        output_file = f"{rootpath}/Average-Merged_{method}.xlsx"
        merge_excel_files_by_method(filelist, output_file)
        print("Done")
def PRImage(saveImagePath):
    legenddic = {
        "EPR.json": "SBR-EPR",
        "SPR.json": "SBR-SPR",
        "SPR/BM25.json": "OSAR-BM25",
        "SPR/EMB.json": "OSAR-ST",
        "SPR/BGE.json": "OSAR-BGE",
        "SPR/Random.json": "OSAR-Random",
        "SPR/LLM/qwen2-70b/BM25.json": "OSAR-LLMs-BM25",
        "SPR/LLM/qwen2-70b/EMB.json": "OSAR-LLMs-ST",
        "SPR/LLM/qwen2-70b/BGE.json": "OSAR-LLMs-BGE",
        "SPR/LLM/qwen2-70b/Random.json": "OSAR-LLMs-Random",
        "SPR/LLM/llama3-70b/BM25.json": "OSAR-LLMs-BM25",
        "SPR/LLM/llama3-70b/EMB.json": "OSAR-LLMs-ST",
        "SPR/LLM/llama3-70b/BGE.json": "OSAR-LLMs-BGE",
        "SPR/LLM/llama3-70b/Random.json": "OSAR-LLMs-Random",
        "BeamSearch/Random.json": "ISAR-Random",
        "BeamSearch/BM25.json": "ISAR-BM25",
        "BeamSearch/EMB.json": "ISAR-ST",
        "BeamSearch/BGE.json": "ISAR-BGE",
        "BeamSearch/LLM/qwen2-70b/Random.json": "ISAR-LLMs-Random",
        "BeamSearch/LLM/qwen2-70b/BM25.json": "ISAR-LLMs-BM25",
        "BeamSearch/LLM/qwen2-70b/EMB.json": "ISAR-LLMs-ST",
        "BeamSearch/LLM/qwen2-70b/BGE.json": "ISAR-LLMs-BGE",
        "BeamSearch/LLM/llama3-70b/Random.json": "ISAR-LLMs-Random",
        "BeamSearch/LLM/llama3-70b/BM25.json": "ISAR-LLMs-BM25",
        "BeamSearch/LLM/llama3-70b/EMB.json": "ISAR-LLMs-ST",
        "BeamSearch/LLM/llama3-70b/BGE.json": "ISAR-LLMs-BGE"
    }
    namedic ={
        "GSR":"SBR",
        "OSR-EEMS":"OSAR-EEMs",
        "ISR-EEMs":"ISAR-EEMs",
    }
    plt.figure(figsize=(24, 14))
    index = 0
    for method in methodlist:
        excelpath = f"{rootpath}/Average-Merged_{method}.xlsx"
        # saveImagePath = f"{rootpath}/Average-Merged_{method}.pdf"
        # Read the Excel file
        df = pd.read_excel(excelpath)
        # Set the Method column as the index
        df.set_index("Method", inplace=True)
        # Create a line plot
        markerlist = ['o', 's', 'D', '^', 'v', 'p', 'P', '*', 'X', 'd']
        # 线性
        linelist = ['-', '--', '-.', ':','-', '--', '-.', ':']
        colorlist = ["#48403E","#252221","#BE0EEF","#A974B8","#A242BC","#25DC50","#7FB58C","#4AB362","#00FF00","#FF4500","#FF6347","#FF00FF"]
        # Iterate over each row and plot the lines
        for i in range(len(df.index)):
            method = df.index[i]
            label = legenddic.get(method, method)  # Map the method name using legenddic
            
            plt.plot(df.columns, df.loc[method], marker=markerlist[index],color=colorlist[index], label=label,linestyle=linelist[index], linewidth=10, markersize=50)
            index +=1
        # Extract numeric parts from x-axis tick labels
        numeric_labels = [str(label).split("@")[1] for label in df.columns]
        # 设置y轴范围
        plt.ylim(0.1, 0.6)
        plt.yticks(np.arange(0.1, 0.41, 0.1))
        plt.tick_params(axis='x', which='major', length=20,labelsize=60)
        plt.tick_params(axis='y', which='major', labelsize=100)
        plt.xticks(ticks=range(len(numeric_labels)), labels=numeric_labels, fontsize=100)

        # Add labels and title
        plt.xlabel("Path Num.", fontsize=100, fontweight='bold')
        plt.ylabel("F1", fontsize=120, fontweight='bold')
        
        # plt.tick_params(axis='y', which='both', length=0, labelsize=100)
        ax = plt.gca()  # 获取当前坐标轴
        # ax.text(0, -0.15, "Pathnum", ha='right', va='center', fontsize=70, fontweight='bold', transform=ax.transAxes)
        # Add legend
    font_properties = FontProperties(weight='bold', size=60)
    plt.legend(prop=font_properties,loc='center',bbox_to_anchor=(0.5, 0.78),ncol=2,labelspacing=0.01, handletextpad=0.01,columnspacing=0.5)

    # Show grid
    # plt.grid(alpha=0.5)
    # Save and close the plot
    # plt.tight_layout()
    plt.subplots_adjust(left=0.2,right=0.97,top=0.9,bottom=0.22)
    # plt.tight_layout()
    plt.savefig(saveImagePath)
    plt.close()

    
if __name__ == '__main__':
    rootpath = "/back-up/gzy/dataset/VLDB/new250/PathRetrieval/Result"
    methodlist = ["GSR","OSR-EEMS","ISR-EEMs"]
    PATHNUM = [1,4,8,16,32]
    mergeExcel()
    for method in methodlist:
        excelpath = f"{rootpath}/Average-Merged_{method}.xlsx"
        saveImagePath = f"{rootpath}/Average-Merged_{method}.pdf"
    saveImagePath = f"{rootpath}/PR-Average-Merged.pdf"
    PRImage(saveImagePath)
    print("Done")