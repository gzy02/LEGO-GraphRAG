
def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("type.type.") or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation or "sameas" in relation:
        return True
output_file= "rel_filter.txt"
input_file= "manual_fb_filter.txt"

f_in = open(input_file)
f_out = open(output_file, "w")
num_line = 0
num_reserve = 0

for line in f_in:
    splitline = line.strip().split("\t")
    num_line += 1
    if len(splitline) < 3:
        continue
    rel = splitline[1]
    if abandon_rels(rel):
        continue
    f_out.write(line)
    if num_line % 1000000 == 0:
        print(num_line, num_reserve)
f_in.close()
f_out.close()
print(num_line, num_reserve)
