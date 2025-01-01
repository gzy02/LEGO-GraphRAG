#!/bin/bash


array=('WebQTest-0' 'WebQTest-1' 'WebQTest-3' 'WebQTest-6' 'WebQTest-7' 'WebQTest-12' 'WebQTest-13' 'WebQTest-14' 'WebQTest-16' 'WebQTest-20' 'WebQTest-21' 'WebQTest-22' 'WebQTest-23' 'WebQTest-24' 'WebQTest-28' 'WebQTest-31' 'WebQTest-32' 'WebQTest-33' 'WebQTest-34' 'WebQTest-35' 'WebQTest-36' 'WebQTest-37' 'WebQTest-38' 'WebQTest-39' 'WebQTest-52' 'WebQTest-54' 'WebQTest-55' 'WebQTest-56' 'WebQTest-58' 'WebQTest-59' 'WebQTest-60' 'WebQTest-61' 'WebQTest-62' 'WebQTest-63' 'WebQTest-64' 'WebQTest-65' 'WebQTest-67' 'WebQTest-68' 'WebQTest-69' 'WebQTest-102' 'WebQTest-103' 'WebQTest-104' 'WebQTest-105' 'WebQTest-106' 'WebQTest-107' 'WebQTest-108' 'WebQTest-109' 'WebQTest-110' 'WebQTest-111' 'WebQTest-112' 'WebQTest-114' 'WebQTest-115' 'WebQTest-116' 'WebQTest-118' 'WebQTest-119' 'WebQTest-134' 'WebQTest-138' 'WebQTest-139' 'WebQTest-141' 'WebQTest-142' 'WebQTest-149' 'WebQTest-150' 'WebQTest-153' 'WebQTest-154' 'WebQTest-155' 'WebQTest-157' 'WebQTest-159' 'WebQTest-160' 'WebQTest-161' 'WebQTest-162' 'WebQTest-163' 'WebQTest-164' 'WebQTest-165' 'WebQTest-166' 'WebQTest-167' 'WebQTest-168' 'WebQTest-169' 'WebQTest-170' 'WebQTest-171' 'WebQTest-172' 'WebQTest-173' 'WebQTest-174' 'WebQTest-175' 'WebQTest-176' 'WebQTest-177' 'WebQTest-178' 'WebQTest-179' 'WebQTest-180' 'WebQTest-181' 'WebQTest-182' 'WebQTest-184' 'WebQTest-185' 'WebQTest-186' 'WebQTest-187' 'WebQTest-188' 'WebQTest-189' 'WebQTest-190' 'WebQTest-191' 'WebQTest-193' 'WebQTest-194')





# 遍历所有键并调用命令
for relation in "${array[@]}"; do
    ./TopPPR -d "$relation" -algo GEN_GROUND_TRUTH -n 1
done
