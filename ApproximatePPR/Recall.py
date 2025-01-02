def getidlist():
    txtPath = f"/back-up/gzy/dataset/VLDB/PPR.txt"
    idlist = []
    with open(txtPath, "r") as f:
        for line in f:
            idlist.append(line.strip())
    return idlist
def ourpprtime():
    alltime = 0
    count = 0
    with open('JsonResult/OurPPRtime.json', 'r', encoding="utf-8") as file:
        data = json.load(file)
        for key, value in data.items():
            if key in idnamelist:
                alltime += value
                count += 1
    print("OurPPR time count:", count)
    return alltime/count


def toppprtime():
    alltime = 0
    count = 0
    idlist = []
    with open("TopPPR/TopPPR_topk.txt", "r") as file:
        for line in file:
            time_value = float(line.split(" ")[1])
            id = line.split(" ")[0]
            if id in idnamelist:
                alltime += time_value
                count += 1
    print("TopPPR time count:", count)
    return alltime/count


def foratime():
    alltime = 0
    count = 0
    with open("fora/fora_topk.txt", "r") as file:
        for line in file:
            id, time_value = line.strip().split()
            if id in idnamelist:
                alltime += float(time_value)
                count += 1
    print("fora time count:", count)
    return alltime/count


def foraplustime():
    alltime = 0
    count = 0
    with open("fora/foraplus_topk.txt", "r") as file:
        for line in file:
            id, time_value = line.strip().split()
            if id in idnamelist:
                alltime += float(time_value)
                count += 1
    print("foraplus time count:", count)
    return alltime/count


def OurPPRrecall(answer, subgraph):
    node = set()
    for sub in subgraph:
        node.add(sub[0])
        node.add(sub[2])
    subgraphnode = list(node)
    nodenum = 0
    for ans in answer:
        if ans in subgraphnode:
            nodenum += 1
    return nodenum/len(answer)
def ToppprRecall(id, answer):
    toppprroot = "TopPPR/Topppr-answer/"
    with open(toppprroot+id+"_output.txt", "r") as file:
        lines = file.readlines()
        idlist = []
        for line in lines[:2000]:
            indexid = line.split(" ")[0]
            idlist.append(indexid)
    countNum = 0
    for ans in answer:
        # ans -> Freebaseid -> indexid
        Freebaseid = name2FreeIDFC(ans)
        try:
            ansid = name2id[Freebaseid]
            ansid = str(int(ansid))
            # print(ansid)

            if ansid in idlist:
                countNum += 1
        except:
            pass
    return countNum/len(answer)


def foraPPRRecall(id, answer):
    forapprroot = "fora/foraresult_2000/"
    with open(forapprroot+id+"_output.txt", "r") as file:
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
            countNum += 1
    return countNum/len(answer)


def OurPPR():
    # 得到Recall
    filepath = f"JsonResult/OurPPR_webqsp_first_200_output.jsonl"
    Recall = 0
    count = 0
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            obj = json.loads(line)
            id = obj["id"]
            if id in idnamelist:
                subgraph = obj["subgraph"]
                try:
                    answer = obj["answers"]
                    Recall += OurPPRrecall(answer, subgraph)
                    count += 1
                except Exception as e:
                    print(e)
    # print("PPR type: OurPPR", "count: ",count)
    ourPPRTime = ourpprtime()
    Recall = Recall/count
    return ["OurPPR", Recall, ourPPRTime]


def TopPPR():
    # 得到Recall
    filepath = f"JsonResult/TopPPR_webqsp_first_200_output.jsonl"
    topPPRRecall = 0
    count = 0
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            obj = json.loads(line)
            id = obj["id"]
            answer = obj["answers"]
            try:
                if id in idnamelist:
                    topPPRRecall += ToppprRecall(id, answer)
                    count += 1
            except:
                pass
    topPPRRecall = topPPRRecall/count
    topPPRTime = toppprtime()
    return ["TopPPR", topPPRRecall, topPPRTime]


def Fora():
    filepath = f"JsonResult/fora_webqsp_first_200_output.jsonl"
    foraRecall = 0
    count = 0
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            obj = json.loads(line)
            id = obj["id"]
            answer = obj["answers"]
            if id in idnamelist:
                try:
                    foraRecall += foraPPRRecall(id, answer)
                    count += 1
                except Exception as e:
                    print(e)
    # 得到Time
    # print("Fora count:",count)
    foraRecall = foraRecall/count
    foraTime = foratime()
    return ["Fora", foraRecall, foraTime]


def Foraplus():
    foraplusRecall = 0
    foraplusTime = 0
    return ["Foraplus", foraplusRecall, foraplusTime]



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
    with open("TopPPR/dataset/name2id.index.json", "r") as fp:
        name2id = json.load(fp)
    with open("/back-up/gzy/id2name.txt", "r") as fp:
        name2FreeID = {}
        for line in fp:
            mid, rel, name = line.strip().split("\t")
            name2FreeID[name] = mid
    # 统计ppr的结果
    spendtime = time.time()-before
    print("processing done! spend time:", spendtime)
    PPRlist = ["OurPPR", "TopPPR", "fora"]
    PPRResultlist = [["PPRtype", "recall", "time"]]

    idnamelist = getidlist()  # 从文件读入idname, 用于统计recall
    for ppr in PPRlist:
        if ppr == "OurPPR":
            PPRResultlist.append(OurPPR())
        elif ppr == "TopPPR":
            PPRResultlist.append(TopPPR())
        elif ppr == "fora":
            PPRResultlist.append(Fora())
    df = pd.DataFrame(PPRResultlist)
    df.to_excel("Recall.xlsx", index=False)
    print("Done!")
