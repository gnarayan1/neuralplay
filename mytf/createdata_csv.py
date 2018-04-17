import csv
import random

def do_some_logic(age, code1, code2):
    if age < 21:
        if code1 < 10:
            return 0
        else:
            return 1
    elif age < 55:
        if code2 < 10:
            return 2
        else:
            return 3
    else:
        return 4

with open('./data/data.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar="'", quoting=csv.QUOTE_MINIMAL)
    for age in range(0,100):
        for code1 in range(0,20):
            for code2 in range(0, 20):
                csvwriter.writerow([age,code1,code2,do_some_logic(age, code1, code2)])


with open('./data/data.csv','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('./data/random_data.csv','w') as target:
    for _, line in data:
        target.write(line)