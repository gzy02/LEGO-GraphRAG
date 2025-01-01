def ourpprtime():
    alltime =0
    count = 0
    with open('JsonResult/OurPPRtime.json', 'r',encoding="utf-8") as file:
        data = json.load(file)
        for key,value in data.items():
            if key in idnamelist:
                alltime+=value
                count +=1
    print("OurPPR time count:",count)
    return alltime/count
def toppprtime():
    alltime =0
    count = 0
    idlist =[]
    with open("TopPPR/TopPPR_topk.txt","r") as file:
        for line in file:
            time_value = float(line.split(" ")[1])
            id = line.split(" ")[0]
            # if time_value <70:
            #     idlist.append(id)
            if id in idnamelist:
                alltime+=time_value
                count+=1
    # print("id list:",idlist)
    # print("len id list:",len(idlist))   
    print("TopPPR time count:",count)
    return alltime/count
def foratime():
    alltime= 0
    count =0 
    with open("fora/fora_topk.txt","r") as file:
        for line in file:
            id,time_value = line.strip().split()
            if id in idnamelist:
                alltime+=float(time_value)
                count+=1
    print("fora time count:",count)
    return alltime/count
def foraplustime():
    alltime= 0
    count =0 
    with open("fora/foraplus_topk.txt","r") as file:
        for line in file:
            id,time_value = line.strip().split()
            if id in idnamelist:
                alltime+=float(time_value)
                count+=1
    print("foraplus time count:",count)
    return alltime/count
def OurPPRrecall(answer,subgraph):
    node = set()
    for sub in subgraph:
        node.add(sub[0])
        node.add(sub[2])
    subgraphnode = list(node)
    nodenum =0
    for ans in answer:
        if ans in subgraphnode:
            nodenum+=1
    return nodenum/len(answer)
# def OurPPRrecall(answer,subgraph):
#     ansset = set()
#     for ans in answer:
#         singleans = ans.split(" ")
#         for single in singleans:
#             for sub in subgraph:
#                 if single in sub[0]:
#                     ansset.add(ans)
#                 if single in sub[2]:
#                     ansset.add(ans)
#     return len(ansset)/len(answer)
def ToppprRecall(id,answer):
    # 思路:1.从Toppper-answer读入txt,获取节点索引
    # 2. 从name2id.index.json读入json获取 Freebase的索引
    # 3. 从id2name.txt 读入txt获得节点名字
    # 获得前2000个节点名字，计算recall
    toppprroot = "TopPPR/Topppr-answer/"
    with open(toppprroot+id+"_output.txt","r") as file:
        lines = file.readlines()
        idlist = []
        for line in lines[:2000]:
            indexid = line.split(" ")[0]
            idlist.append(indexid)
    countNum = 0
    # print(idlist)
    for ans in answer:
        # ans -> Freebaseid -> indexid
        Freebaseid = name2FreeIDFC(ans)
        try:
            ansid = name2id[Freebaseid]
            ansid = str(int(ansid))
            # print(ansid)

            if ansid in idlist:
                countNum+=1
        except:
            pass
    return countNum/len(answer)
def foraPPRRecall(id,answer):
    forapprroot = "fora/foraresult_2000/"
    # with open(forapprroot+id+"_output.txt","r") as file:
    #     lines = file.readlines()
    #     namelist = []
    #     for line in lines[:2000]:
    #         indexid = line.split(" ")[0]
    #         # if indexid=='0':
    #         #     continue
    #         Freebaseid = id2FreeBase[indexid]
    #         name = id2nameFc(Freebaseid)
    #         namelist.append(name)
    # # 根据namelist 和answer算recall
    # countNum =0
    with open(forapprroot+id+"_output.txt","r") as file:
        lines = file.readlines()
        idlist = []
        for line in lines[:2000]:
            indexid = line.split(" ")[0]
            idlist.append(indexid)
    countNum = 0
    # print(idlist)
    for ans in answer:
        # ans -> Freebaseid -> indexid
        Freebaseid = name2FreeIDFC(ans)
        ansid = name2id[Freebaseid]
        ansid = str(int(ansid))
        # print(id,ansid)
        if ansid in idlist:
            countNum+=1
    return countNum/len(answer)
# def foraPlusPPRRecall(id,answer):
#     forapprroot = "fora/foraresultplus_2000/"
#     with open(forapprroot+id+"_output.txt","r") as file:
#         lines = file.readlines()
#         namelist = []
#         for line in lines[:2000]:
#             indexid = line.split(" ")[0]
#             Freebaseid = id2FreeBase[indexid]
#             name = id2nameFc(Freebaseid)
#             namelist.append(name)
#     # 根据namelist 和answer算recall
#     countNum =0
#     for ans in answer:
#         if ans in namelist:
#             countNum+=1
#     return countNum/len(answer)
def OurPPR():
    # 得到Recall
    filepath = f"JsonResult/OurPPR_webqsp_first_200_output.jsonl"
    Recall = 0
    count =0
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            obj = json.loads(line)
            id = obj["id"]
            if id in idnamelist:
                subgraph = obj["subgraph"]
                try:
                    answer = obj["answers"]
                    Recall += OurPPRrecall(answer,subgraph)
                    count += 1
                except Exception as e:
                    print(e)
    # print("PPR type: OurPPR", "count: ",count)
    ourPPRTime = ourpprtime()
    Recall = Recall/count
    return ["OurPPR",Recall,ourPPRTime]
def TopPPR():
    # 得到Recall
    filepath = f"JsonResult/TopPPR_webqsp_first_200_output.jsonl"
    topPPRRecall = 0
    count =0
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            obj = json.loads(line)
            id = obj["id"]
            answer = obj["answers"]
            try:
                if id in idnamelist:
                    topPPRRecall += ToppprRecall(id,answer)
                    count += 1  
            except:
                pass
    # 得到Time
    # print("TopPPR count:",count)
    topPPRRecall= topPPRRecall/count
    topPPRTime =toppprtime()
    return ["TopPPR",topPPRRecall,topPPRTime]
def Fora():
    # 得到Recall
    # 得到Time
    filepath = f"JsonResult/fora_webqsp_first_200_output.jsonl"
    foraRecall = 0
    count =0
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            obj = json.loads(line)
            id = obj["id"]
            answer = obj["answers"]
            if id in idnamelist:
                try:
                    foraRecall += foraPPRRecall(id,answer)
                    count += 1
                except Exception as e:
                    print(e)
    # 得到Time
    # print("Fora count:",count)
    foraRecall= foraRecall/count
    foraTime =foratime()
    return ["Fora",foraRecall,foraTime]
# def ForaPlus():
#     # 得到Recall
#     # 得到Time
#     filepath = f"JsonResult/fora_webqsp_first_200_output.jsonl"
#     foraRecall = 0
#     count =0
#     with open(filepath, "r") as fp:
#         lines = fp.readlines()
#         for line in lines:
#             obj = json.loads(line)
#             id = obj["id"]
#             answer = obj["answers"]
#             if id in idnamelist:
#                 foraRecall += foraPlusPPRRecall(id,answer)
#                 count += 1  
#     # 得到Time
#     print("Fora count:",count)
#     foraRecall= foraRecall/count
#     foraTime =foraplustime()
#     return ["Fora+",foraRecall,foraTime]
def Foraplus():
    foraplusRecall =0
    foraplusTime =0
    return ["Foraplus",foraplusRecall,foraplusTime]
# def id2nameFc(mid):
#     if mid in id2name:
#         return id2name[mid]
#     else:
#         return mid
def name2FreeIDFC(name):
    if name in name2FreeID:
        return name2FreeID[name]
    else:
        return name
if __name__ == "__main__":
    import json
    import pandas as pd
    import time
    before = time.time()
    with open("TopPPR/dataset/name2id.index.json","r") as fp:
        name2id = json.load(fp)
    # id2FreeBase = {str(v): str(k) for k, v in name2id.items()}
    # 读入id2name.txt
    with open("/back-up/gzy/id2name.txt","r") as fp:
        name2FreeID = {}
        for line in fp:
            mid,rel,name = line.strip().split("\t")
            name2FreeID[name] = mid
    # 统计ppr的结果
    spendtime = time.time()-before
    print("processing done! spend time:",spendtime)
    PPRlist = ["OurPPR","TopPPR","fora"]
    PPRResultlist = [["PPRtype","recall","time"]]
    #idnamelist = ['WebQTest-24', 'WebQTest-116', 'WebQTest-52', 'WebQTest-115', 'WebQTest-114', 'WebQTest-54', 'WebQTest-21', 'WebQTest-55', 'WebQTest-20', 'WebQTest-112', 'WebQTest-56', 'WebQTest-23', 'WebQTest-111', 'WebQTest-22', 'WebQTest-110', 'WebQTest-58', 'WebQTest-59', 'WebQTest-28', 'WebQTest-119', 'WebQTest-118', 'WebQTest-32', 'WebQTest-104', 'WebQTest-33', 'WebQTest-105', 'WebQTest-7', 'WebQTest-106', 'WebQTest-6', 'WebQTest-31', 'WebQTest-107', 'WebQTest-36', 'WebQTest-1', 'WebQTest-37', 'WebQTest-0', 'WebQTest-34', 'WebQTest-3', 'WebQTest-102', 'WebQTest-35', 'WebQTest-103', 'WebQTest-38', 'WebQTest-39', 'WebQTest-108', 'WebQTest-109', 'WebQTest-243', 'WebQTest-171', 'WebQTest-170', 'WebQTest-241', 'WebQTest-173', 'WebQTest-240', 'WebQTest-172', 'WebQTest-247', 'WebQTest-175', 'WebQTest-246', 'WebQTest-174', 'WebQTest-245', 'WebQTest-199', 'WebQTest-177', 'WebQTest-176', 'WebQTest-197', 'WebQTest-179', 'WebQTest-196', 'WebQTest-178', 'WebQTest-195', 'WebQTest-194', 'WebQTest-193', 'WebQTest-191', 'WebQTest-190', 'WebQTest-225', 'WebQTest-224', 'WebQTest-227', 'WebQTest-226', 'WebQTest-221', 'WebQTest-188', 'WebQTest-220', 'WebQTest-189', 'WebQTest-222', 'WebQTest-184', 'WebQTest-185', 'WebQTest-186', 'WebQTest-187', 'WebQTest-180', 'WebQTest-181', 'WebQTest-182', 'WebQTest-232', 'WebQTest-233', 'WebQTest-231', 'WebQTest-237', 'WebQTest-234', 'WebQTest-168', 'WebQTest-235', 'WebQTest-169', 'WebQTest-166', 'WebQTest-14', 'WebQTest-167', 'WebQTest-164', 'WebQTest-16', 'WebQTest-239', 'WebQTest-165', 'WebQTest-162', 'WebQTest-163', 'WebQTest-160', 'WebQTest-12', 'WebQTest-161', 'WebQTest-13', 'WebQTest-69', 'WebQTest-68', 'WebQTest-159', 'WebQTest-61', 'WebQTest-153', 'WebQTest-60', 'WebQTest-63', 'WebQTest-62', 'WebQTest-150', 'WebQTest-65', 'WebQTest-157', 'WebQTest-64', 'WebQTest-67', 'WebQTest-155', 'WebQTest-154', 'WebQTest-149', 'WebQTest-139', 'WebQTest-138', 'WebQTest-207', 'WebQTest-206', 'WebQTest-141', 'WebQTest-134', 'WebQTest-205', 'WebQTest-142', 'WebQTest-204', 'WebQTest-131', 'WebQTest-202', 'WebQTest-145', 'WebQTest-201', 'WebQTest-133', 'WebQTest-147', 'WebQTest-132', 'WebQTest-43', 'WebQTest-42', 'WebQTest-41', 'WebQTest-128', 'WebQTest-129', 'WebQTest-47', 'WebQTest-218', 'WebQTest-46', 'WebQTest-45', 'WebQTest-44', 'WebQTest-214', 'WebQTest-122', 'WebQTest-215', 'WebQTest-123', 'WebQTest-49', 'WebQTest-216', 'WebQTest-48', 'WebQTest-217', 'WebQTest-121', 'WebQTest-210', 'WebQTest-126', 'WebQTest-211', 'WebQTest-127', 'WebQTest-212', 'WebQTest-124', 'WebQTest-213', 'WebQTest-125']
    #idnamelist = ['WebQTest-0', 'WebQTest-1', 'WebQTest-3', 'WebQTest-6', 'WebQTest-7', 'WebQTest-12', 'WebQTest-13', 'WebQTest-14', 'WebQTest-16', 'WebQTest-20', 'WebQTest-21', 'WebQTest-22', 'WebQTest-23', 'WebQTest-24', 'WebQTest-28', 'WebQTest-31', 'WebQTest-32', 'WebQTest-33', 'WebQTest-34', 'WebQTest-35', 'WebQTest-36', 'WebQTest-37', 'WebQTest-38', 'WebQTest-39', 'WebQTest-52', 'WebQTest-54', 'WebQTest-55', 'WebQTest-56', 'WebQTest-58', 'WebQTest-59', 'WebQTest-60', 'WebQTest-61', 'WebQTest-62', 'WebQTest-63', 'WebQTest-64', 'WebQTest-65', 'WebQTest-67', 'WebQTest-68', 'WebQTest-69', 'WebQTest-102', 'WebQTest-103', 'WebQTest-104', 'WebQTest-105', 'WebQTest-106', 'WebQTest-107', 'WebQTest-108', 'WebQTest-109', 'WebQTest-110', 'WebQTest-111', 'WebQTest-112', 'WebQTest-114', 'WebQTest-115', 'WebQTest-116', 'WebQTest-118', 'WebQTest-119', 'WebQTest-134', 'WebQTest-138', 'WebQTest-139', 'WebQTest-141', 'WebQTest-142', 'WebQTest-149', 'WebQTest-150', 'WebQTest-153', 'WebQTest-154', 'WebQTest-155', 'WebQTest-157', 'WebQTest-159', 'WebQTest-160', 'WebQTest-161', 'WebQTest-162', 'WebQTest-163', 'WebQTest-164', 'WebQTest-165', 'WebQTest-166', 'WebQTest-167', 'WebQTest-168', 'WebQTest-169', 'WebQTest-170', 'WebQTest-171', 'WebQTest-172', 'WebQTest-173', 'WebQTest-174', 'WebQTest-175', 'WebQTest-176', 'WebQTest-177', 'WebQTest-178', 'WebQTest-179', 'WebQTest-180', 'WebQTest-181', 'WebQTest-182', 'WebQTest-184', 'WebQTest-185', 'WebQTest-186', 'WebQTest-187', 'WebQTest-188', 'WebQTest-189', 'WebQTest-190', 'WebQTest-191', 'WebQTest-193', 'WebQTest-194']
    #idnamelist = ['WebQTest-111', 'WebQTest-23','WebQTest-33']
    idnamelist = ['WebQTest-115', 'WebQTest-114', 'WebQTest-54', 'WebQTest-21', 'WebQTest-23', 'WebQTest-111', 'WebQTest-110', 'WebQTest-58', 'WebQTest-59', 'WebQTest-28', 'WebQTest-104', 'WebQTest-33', 'WebQTest-7', 'WebQTest-106', 'WebQTest-6', 'WebQTest-31', 'WebQTest-107', 'WebQTest-36', 'WebQTest-1', 'WebQTest-3', 'WebQTest-102', 'WebQTest-35', 'WebQTest-103', 'WebQTest-38', 'WebQTest-109', 'WebQTest-243', 'WebQTest-170', 'WebQTest-241', 'WebQTest-173', 'WebQTest-247', 'WebQTest-175', 'WebQTest-174', 'WebQTest-245', 'WebQTest-179', 'WebQTest-196', 'WebQTest-195', 'WebQTest-194', 'WebQTest-227', 'WebQTest-221', 'WebQTest-220', 'WebQTest-187', 'WebQTest-180', 'WebQTest-232', 'WebQTest-233', 'WebQTest-231', 'WebQTest-237', 'WebQTest-168', 'WebQTest-235', 'WebQTest-164', 'WebQTest-16', 'WebQTest-165', 'WebQTest-160', 'WebQTest-12', 'WebQTest-161', 'WebQTest-13', 'WebQTest-69', 'WebQTest-68', 'WebQTest-63', 'WebQTest-149', 'WebQTest-139', 'WebQTest-138', 'WebQTest-206', 'WebQTest-141', 'WebQTest-142', 'WebQTest-202', 'WebQTest-133', 'WebQTest-132', 'WebQTest-41', 'WebQTest-128', 'WebQTest-129', 'WebQTest-47', 'WebQTest-218', 'WebQTest-46', 'WebQTest-45', 'WebQTest-214', 'WebQTest-122', 'WebQTest-215', 'WebQTest-121', 'WebQTest-210', 'WebQTest-126', 'WebQTest-211', 'WebQTest-127', 'WebQTest-212']
    for ppr in PPRlist:
        if ppr =="OurPPR":
            PPRResultlist.append(OurPPR())
        elif ppr =="TopPPR":
            PPRResultlist.append(TopPPR())
        elif ppr =="fora":
            PPRResultlist.append(Fora())
        # elif ppr =="foraplus":
        #     PPRResultlist.append(Foraplus())
    df = pd.DataFrame(PPRResultlist)
    df.to_excel("Recall.xlsx",index=False)
    print("Done!")