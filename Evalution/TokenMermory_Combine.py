# import pandas as pd
# import pandas as pd

# def merge_tables(A_path, B_path, output_path):
#     # 读取 A 和 B 表格
#     A_df = pd.read_excel(A_path)
#     B_df = pd.read_excel(B_path)

#     # 获取 Instance 列的名称
#     instance_col = 'Instance'
    
#     # 创建一个新的 B 表格副本
#     result_df = B_df.copy()

#     # 遍历 B 表格的每一行
#     for i, b_row in B_df.iterrows():
#         # 获取 B 表格当前行的 Instance 名称，拆分并替换
#         b_instance = b_row[instance_col].split("+")[0]
#         b_instance = b_instance.replace("SBE-PPR", "SBE")
        
#         # 在 A 表格中查找对应的行
#         matching_row = A_df[A_df[instance_col].str.contains(b_instance, case=False, na=False)]
        
#         if not matching_row.empty:
#             # 获取 A 表格对应行的数据（除 Instance 列外）
#             a_row = matching_row.iloc[0].drop(instance_col)
#             # 将 A 表格的数据添加到 B 表格的对应行
#             result_df.loc[i, a_row.index] += a_row.values
    
#     # 将结果保存到新的 Excel 文件
#     result_df.to_excel(output_path, index=False)
#     print(f"合并后的文件已保存到: {output_path}")

# datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
# # datasetlist = ["GrailQA"]

# tasklist = ["Token", "Memory"]
# taskinfo = {
#     "Token": [f"EEMs Token",f"LLMs Token"],
#     "Memory": [f"PeakMemory",f"AveageMemory"]
# }
# for task in tasklist:
#     for dataset in datasetlist:
#         merge_tables(
#             f'/back-up/gzy/dataset/VLDB/SpendExp/{dataset}-{task}@32_SE.xlsx', 
#             f'/back-up/gzy/dataset/VLDB/SpendExp/{dataset}-{task}@32_PR.xlsx', 
#             f'/back-up/gzy/dataset/VLDB/SpendExp/{dataset}-{task}.xlsx',
#         )
import pandas as pd
import pandas as pd

def merge_tables(A_path, B_path, output_path):
    # 读取 A 和 B 表格
    A_df = pd.read_excel(A_path)
    B_df = pd.read_excel(B_path)

    # 获取 Instance 列的名称
    instance_col = 'Instance'
    
    # 创建一个新的 B 表格副本
    result_df = B_df.copy()

    # 遍历 B 表格的每一行
    for i, b_row in B_df.iterrows():
        # 获取 B 表格当前行的 Instance 名称，拆分并替换
        b_instance = b_row[instance_col].split("+")[0]
        b_instance = b_instance.replace("SBE-PPR", "SBE")
        
        # 在 A 表格中查找对应的行
        matching_row = A_df[A_df[instance_col].str.contains(b_instance, case=False, na=False)]
        
        if not matching_row.empty:
            # 获取 A 表格对应行的数据（除 Instance 列外）
            a_row = matching_row.iloc[0].drop(instance_col)
            # 将 A 表格的数据添加到 B 表格的对应行
            result_df.loc[i, a_row.index] += a_row.values
    # 遍历result_df的每一行，如果instance名称包含LLM,就把EEMs Token列的值置零
    # for i, row in result_df.iterrows():
    #     if "LLM" in row[instance_col]:
    #         result_df.at[i, taskinfo["Token"][0]] = 0
    # 将结果保存到新的 Excel 文件
    result_df.to_excel(output_path, index=False)
    print(f"合并后的文件已保存到: {output_path}")

datasetlist = ["CWQ","webqsp","GrailQA","WebQuestion"]
# datasetlist = ["GrailQA"]

tasklist = ["Token"]
taskinfo = {
    "Token": [f"EEMs Token",f"LLMs Token"],
    "Memory": [f"PeakMemory",f"AveageMemory"]
}
for task in tasklist:
    for dataset in datasetlist:
        merge_tables(
            f'/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-{task}@32_SE.xlsx', 
            f'/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-{task}@32_PR.xlsx', 
            f'/back-up/gzy/dataset/VLDB/Pipeline/Generation/SpendExp/{dataset}-{task}.xlsx',
        )