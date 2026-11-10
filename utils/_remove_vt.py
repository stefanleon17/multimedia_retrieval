import re
import os

def remove_vt(filename):
    temp = filename + '.tmp'

    with open(filename, 'r') as fin, open(temp, 'w') as fout:
        for line in fin:
            if line.lstrip().startswith('vt'):
                continue

            line = re.sub(r'/.*?/', '//', line)
            fout.write(line)

    os.replace(temp, filename)


remove_vt("C:\\Users\\stefa\\Downloads\\sphere.obj")