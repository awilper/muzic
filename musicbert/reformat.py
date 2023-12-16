import glob
import os

train_files = glob.glob("hotnessmeta_data_raw/*/train.label")
test_files = glob.glob("hotnessmeta_data_raw/*/test.label")
for file in train_files + test_files:
    f = open(file)
    content = f.read()
    new_content = ''
    lines = content.split("\n")
    for line in lines:
        newline = "".join(line.split(' '))
        new_content += newline + "\n"
    f.close()
    w = open(file, 'w')
    w.write(new_content)
    w.close()

