input_file = "data/fb_en.txt"
output_file = "process_data/id2name.txt"

f_out = open(output_file, "w")

relation = "type.object.name"
with open(input_file, "r") as fp:
    for line in fp.readlines():
        split_line = line.strip().split("\t")
        if split_line[1] == relation:
            f_out.write(line)
f_out.close()
