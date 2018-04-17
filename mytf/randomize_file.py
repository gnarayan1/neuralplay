import random
with open('the_file','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('another_file','w') as target:
    for _, line in data:
        target.write( line )