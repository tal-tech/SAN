import os
from tqdm import tqdm


class Tree:
    def __init__(self, label, parent_label='None', id=0, parent_id=0, op='none'):
        self.children = []
        self.label = label
        self.id = id
        self.parent_id = parent_id
        self.parent_label = parent_label
        self.op = op


def convert(root: Tree, f):
    if root.tag == 'N-T':
        f.write(f'{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n')
        for child in root.children:
            convert(child, f)
    else:
        f.write(f'{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n')



label = '../train_latex.txt'
out = 'train_hyb'

position = set(['^', '_'])
math = set(['\\frac','\sqrt'])

with open(label) as f:
    lines = f.readlines()
num = 0
for line in tqdm(lines):
    # line = 'RIT_2014_178.jpg x ^ { \\frac { p } { q } } = \sqrt [ q ] { x ^ { p } } = \sqrt [ q ] { x ^ { p } }'
    name, *words = line.split()
    name = name.split('.')[0]

    parents = []
    root = Tree('root', parent_label='root', parent_id=-1)

    struct_list = ['\\frac', '\sqrt']

    labels = []
    id = 1
    parents = [Tree('<sos>', id=0)]
    parent = Tree('<sos>', id=0)

    for i in range(len(words)):
        a = words[i]
        if a == '\\limits':
            continue
        if i == 0 and words[i] in ['_', '^', '{', '}']:
            print(name)
            break

        elif words[i] == '{':
            if words[i-1] == '\\frac':
                labels.append([id, 'struct', parent.id, parent.label])
                parents.append(Tree('\\frac', id=parent.id, op='above'))
                id += 1
                parent = Tree('above', id=parents[-1].id+1)
            elif words[i-1] == '}' and parents[-1].label == '\\frac' and parents[-1].op == 'above':
                parent = Tree('below', id=parents[-1].id+1)
                parents[-1].op = 'below'

            elif words[i-1] == '\sqrt':
                labels.append([id, 'struct', parent.id, '\sqrt'])
                parents.append(Tree('\sqrt', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i-1] == ']' and parents[-1].label == '\sqrt':
                parent = Tree('inside', id=parents[-1].id+1)

            elif words[i-1] == '^':
                if words[i-2] != '}':
                    if words[i-2] == '\sum':
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree('\sum', id=parent.id))
                        parent = Tree('above', id=id)
                        id += 1

                    else:
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree(words[i-2], id=parent.id))
                        parent = Tree('sup', id=id)
                        id += 1

                else:
                    # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                    if parents[-1].label == '\sum':
                        parent = Tree('above', id=parents[-1].id+1)
                    else:
                        parent = Tree('sup', id=parents[-1].id + 1)
                    # id += 1

            elif words[i-1] == '_':
                if words[i-2] != '}':
                    if words[i-2] == '\sum':
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree('\sum', id=parent.id))
                        parent = Tree('below', id=id)
                        id += 1

                    else:
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree(words[i-2], id=parent.id))
                        parent = Tree('sub', id=id)
                        id += 1

                else:
                    # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                    if parents[-1].label == '\sum':
                        parent = Tree('below', id=parents[-1].id+1)
                    else:
                        parent = Tree('above', id=parents[-1].id+1)
                    # id += 1
            else:
                print('unknown word before {', name, i)


        elif words[i] == '[' and words[i-1] == '\sqrt':
            labels.append([id, 'struct', parent.id, '\sqrt'])
            parents.append(Tree('\sqrt', id=parent.id))
            parent = Tree('L-sup', id=id)
            id += 1
        elif words[i] == ']' and parents[-1].label == '\sqrt':
            labels.append([id, '<eos>', parent.id, parent.label])
            id += 1

        elif words[i] == '}':

            if words[i-1] != '}':
                labels.append([id, '<eos>', parent.id, parent.label])
                id += 1

            if i + 1 < len(words) and words[i+1] == '{' and parents[-1].label == '\\frac' and parents[-1].op == 'above':
                continue
            if i + 1 < len(words) and words[i + 1] in ['_', '^']:
                continue
            elif i + 1 < len(words) and words[i + 1] != '}':
                parent = Tree('right', id=parents[-1].id + 1)

            parents.pop()


        else:
            if words[i] in ['^', '_']:
                continue
            labels.append([id, words[i], parent.id, parent.label])
            parent = Tree(words[i],id=id)
            id += 1


    parent_dict = {0:[]}
    for i in range(len(labels)):
        parent_dict[i+1] = []
        parent_dict[labels[i][2]].append(labels[i][3])

    with open(f'train_hyb/{name}.txt', 'w') as f:
        for line in labels:
            id, label, parent_id, parent_label = line
            if label != 'struct':
                f.write(f'{id}\t{label}\t{parent_id}\t{parent_label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n')
            else:
                tem = f'{id}\t{label}\t{parent_id}\t{parent_label}'
                tem = tem + '\tabove' if 'above' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tbelow' if 'below' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tsub' if 'sub' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tsup' if 'sup' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tL-sup' if 'L-sup' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tinside' if 'inside' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tright' if 'right' in parent_dict[id] else tem + '\tNone'
                f.write(tem + '\n')
        if label != '<eos>':
            f.write(f'{id+1}\t<eos>\t{id}\t{label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n')







