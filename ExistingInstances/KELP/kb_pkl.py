import pickle

# 读取 kb.txt 文件内容
with open("/back-up/gzy/rel_filter.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()

id2name_dict = {}
with open("/back-up/gzy/id2name.txt", 'r', encoding='utf-8') as fp:
    for line in fp:
        mid, rel, name = line.strip().split('\t')
        id2name_dict[mid] = name


def id2name(mid):
    if mid in id2name_dict:
        return id2name_dict[mid]
    else:
        return mid


kg = {}
for line in lines:
    entity, relation, obj = line.strip().split('\t')
    entity = id2name(entity)
    obj = id2name(obj)
    if entity not in kg:
        kg[entity] = {}
    if relation not in kg[entity]:
        kg[entity][relation] = []
    kg[entity][relation].append(obj)

# 将字典数据结构保存为 Pickle 文件
with open('freebase_kg.pickle', 'wb') as file:
    pickle.dump(kg, file)
