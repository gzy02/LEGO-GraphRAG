#!/bin/bash

# 定义 n 和 k 的值
declare -A k_values


k_values=(
    ["WebQTest-0"]=519620
    ["WebQTest-1"]=2334910
    ["WebQTest-100"]=861647
    ["WebQTest-102"]=680975
    ["WebQTest-103"]=2575500
    ["WebQTest-104"]=2681175
    ["WebQTest-105"]=95508
    ["WebQTest-106"]=1107163
    ["WebQTest-107"]=2671736
    ["WebQTest-108"]=1893019
    ["WebQTest-109"]=2319241
    ["WebQTest-110"]=2383600
    ["WebQTest-111"]=132480
    ["WebQTest-112"]=2104202
    ["WebQTest-114"]=2582375
    ["WebQTest-115"]=2383600
    ["WebQTest-116"]=2354942
    ["WebQTest-118"]=40598
    ["WebQTest-119"]=1902085
    ["WebQTest-12"]=1739173
    ["WebQTest-121"]=2030984
    ["WebQTest-122"]=2118727
    ["WebQTest-123"]=532346
    ["WebQTest-124"]=242410
    ["WebQTest-125"]=1729618
    ["WebQTest-126"]=2393
    ["WebQTest-127"]=1679614
    ["WebQTest-128"]=2570837
    ["WebQTest-129"]=1652707
    ["WebQTest-13"]=2357816
    ["WebQTest-131"]=2373636
    ["WebQTest-132"]=2520417
    ["WebQTest-133"]=2510247
    ["WebQTest-134"]=1594790
    ["WebQTest-138"]=1539799
    ["WebQTest-139"]=6450
    ["WebQTest-14"]=64084
    ["WebQTest-141"]=2571765
    ["WebQTest-142"]=303
    ["WebQTest-145"]=2425876
    ["WebQTest-146"]=361659
    ["WebQTest-147"]=1663156
    ["WebQTest-149"]=922104
    ["WebQTest-150"]=479610
    ["WebQTest-153"]=1281012
    ["WebQTest-154"]=2919568
    ["WebQTest-155"]=2489874
    ["WebQTest-156"]=1975439
    ["WebQTest-157"]=31914
    ["WebQTest-159"]=776394
    ["WebQTest-16"]=3661249
    ["WebQTest-160"]=1764377
    ["WebQTest-161"]=1886064
    ["WebQTest-162"]=1030121
    ["WebQTest-163"]=105858
    ["WebQTest-164"]=1895420
    ["WebQTest-165"]=665536
    ["WebQTest-166"]=2698300
    ["WebQTest-167"]=855410
    ["WebQTest-168"]=2405128
    ["WebQTest-169"]=434467
    ["WebQTest-170"]=3532129
    ["WebQTest-171"]=1088443
    ["WebQTest-172"]=783689
    ["WebQTest-173"]=37835
    ["WebQTest-174"]=2281584
    ["WebQTest-175"]=2284056
    ["WebQTest-176"]=1562185
    ["WebQTest-177"]=2337600
    ["WebQTest-178"]=260670
    ["WebQTest-179"]=1801103
    ["WebQTest-180"]=845497
    ["WebQTest-181"]=95508
    ["WebQTest-182"]=1608145
    ["WebQTest-184"]=2307356
    ["WebQTest-185"]=1961444
    ["WebQTest-186"]=381098
    ["WebQTest-187"]=1895420
    ["WebQTest-188"]=1322575
    ["WebQTest-189"]=242410
    ["WebQTest-19"]=1614949
    ["WebQTest-190"]=2698300
    ["WebQTest-191"]=2447324
    ["WebQTest-193"]=2598485
    ["WebQTest-194"]=364113
    ["WebQTest-195"]=3019906
    ["WebQTest-196"]=412977
    ["WebQTest-197"]=899866
    ["WebQTest-199"]=55828
    ["WebQTest-20"]=1764779
    ["WebQTest-200"]=165711
    ["WebQTest-201"]=2329438
    ["WebQTest-202"]=1614493
    ["WebQTest-204"]=242410
    ["WebQTest-205"]=1932491
    ["WebQTest-206"]=2546642
    ["WebQTest-207"]=856948
    ["WebQTest-209"]=368094
    ["WebQTest-21"]=1898154
    ["WebQTest-210"]=2904431
    ["WebQTest-211"]=1614493
    ["WebQTest-212"]=909994
    ["WebQTest-213"]=2053710
    ["WebQTest-214"]=34945
    ["WebQTest-215"]=3018157
    ["WebQTest-216"]=16000
    ["WebQTest-217"]=14685590
    ["WebQTest-218"]=2079037
    ["WebQTest-22"]=242410
    ["WebQTest-220"]=198502
    ["WebQTest-221"]=2303837
    ["WebQTest-222"]=367164
    ["WebQTest-224"]=2333371
    ["WebQTest-225"]=810371
    ["WebQTest-226"]=1856963
    ["WebQTest-227"]=368886
    ["WebQTest-23"]=2657060
    ["WebQTest-231"]=144356
    ["WebQTest-232"]=1368696
    ["WebQTest-233"]=1113772
    ["WebQTest-234"]=114978
    ["WebQTest-235"]=2404614
    ["WebQTest-237"]=2665624
    ["WebQTest-239"]=1499076
    ["WebQTest-24"]=2693999
    ["WebQTest-240"]=2425876
    ["WebQTest-241"]=1239327
    ["WebQTest-243"]=2489874
    ["WebQTest-245"]=12504
    ["WebQTest-246"]=2309539
    ["WebQTest-247"]=1267781
    ["WebQTest-26"]=1920709
    ["WebQTest-28"]=751874
    ["WebQTest-3"]=713364
    ["WebQTest-31"]=2265731
    ["WebQTest-32"]=93748
    ["WebQTest-33"]=556221
    ["WebQTest-34"]=2298473
    ["WebQTest-35"]=1087619
    ["WebQTest-36"]=1759953
    ["WebQTest-37"]=1586530
    ["WebQTest-38"]=2489874
    ["WebQTest-39"]=1888275
    ["WebQTest-41"]=9572
    ["WebQTest-42"]=2354942
    ["WebQTest-43"]=1823901
    ["WebQTest-44"]=838428
    ["WebQTest-45"]=1361046
    ["WebQTest-46"]=2345410
    ["WebQTest-47"]=2702184
    ["WebQTest-48"]=481322
    ["WebQTest-49"]=123501
    ["WebQTest-51"]=2570849
    ["WebQTest-52"]=2653511
    ["WebQTest-54"]=1857578
    ["WebQTest-55"]=3040293
    ["WebQTest-56"]=1720417
    ["WebQTest-58"]=2319883
    ["WebQTest-59"]=783819
    ["WebQTest-6"]=2575213
    ["WebQTest-60"]=1937158
    ["WebQTest-61"]=424199
    ["WebQTest-62"]=2281349
    ["WebQTest-63"]=2372119
    ["WebQTest-64"]=2243164
    ["WebQTest-65"]=205926
    ["WebQTest-66"]=1708756
    ["WebQTest-67"]=2608214
    ["WebQTest-68"]=1909873
    ["WebQTest-69"]=404901
    ["WebQTest-7"]=2276948
)


# 输出文件
output_file="TopPPR_time_All.txt"

# 清空输出文件（如果已存在）
> "$output_file"

# 循环遍历所有的 relations
for relation in "${!k_values[@]}"; do
    # 计算执行时间并提取实际时间
    start_time=$(date +%s%N)  # 获取当前时间的纳秒数
    ./TopPPR -d "$relation" -algo TopPPR -r 1 -n 1 -k 1000
    end_time=$(date +%s%N)  # 获取结束时间的纳秒数

    # 计算花费的时间（纳秒转换为秒）
    elapsed_time=$((($end_time - $start_time) / 1000000000))

    # 将 relation 和时间写入文件
    echo "$relation $elapsed_time" >> "$output_file"
done