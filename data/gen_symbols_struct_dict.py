import os
import glob
from tqdm import tqdm

label_path = 'test-bak'

labels = glob.glob(os.path.join(label_path, '*.txt'))

words_dict = set(['<eos>', '<sos>', 'struct'])

with open('word.txt', 'w') as writer:
    writer.write('<eos>\n<sos>\nstruct\n')
    i = 3
    for item in tqdm(labels):
        with open(item) as f:
            lines = f.readlines()
        for line in lines:
            cid, c, pid, p, *r = line.strip().split()
            if c not in words_dict:
                words_dict.add(c)
                writer.write(f'{c}\n')
                i+=1
    writer.write('above\nbelow\nsub\nsup\nl_sup\ninside\nright')
print(i)


