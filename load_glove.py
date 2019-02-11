import numpy as np

def load_glove_embeddings(glove_file):
    f = open(glove_file,'r')
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    f.close()
    return model

def load_wiki_embeddings(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 1
    for line in fin:
        tokens = line.rstrip().split(' ')
        #data[tokens[0]] = map(float, tokens[1:])
        data[tokens[0]] = np.array([float(val) for val in tokens[1:]]) 
        if(i >= 400000):
            break
        i += 1
    print("Done.",len(data)," words loaded!")
    return data
