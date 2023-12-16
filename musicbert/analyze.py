import glob
import os
import math

train_files = glob.glob("hotness_data_raw/*/train.label")
test_files = glob.glob("hotness_data_raw/*/test.label")


train_buckets = [0]*11
test_buckets = [0]*11

for file in train_files:
    f = open(file)
    content = f.read().split("\n")
    for line in content:
        if not line:
            continue
        score = int(float(line.strip())*10)
        train_buckets[score] += 1
    f.close()

for file in test_files:
    f = open(file)
    content = f.read().split("\n")
    for line in content:
        if not line:
            continue
        score = int(float(line.strip())*10)
        test_buckets[score] += 1
    f.close()

print(train_buckets)
print(test_buckets)
