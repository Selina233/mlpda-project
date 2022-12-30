'''
这是一些功能函数。
为了方便，我把一些单词处理的函数搬到这里来了。
'''
import os
import re

import numpy as np
import tqdm


def get_glove_encoding(glove_data_path: str = 'data/glove.6B.100d.txt'):
    '''
    获得glove里面的单词编码。
    
    返回：word2index（单词映射到单词id的字典），
        vocabulary_vectors（每个单词的embedding vector组成的列表，用单词的id作为下标来访问）
    '''
    # 读取文本文件
    glove_data = open(glove_data_path, encoding='utf-8')
    
    # 把句子转成id向量
    word_list = []
    vocabulary_vectors = []
    for line in glove_data.readlines():
        temp = line.strip('\n').split(' ')  # 一个列表
        name = temp[0]
        word_list.append(name.lower())
        vector = [temp[i] for i in range(1, len(temp))]  # 向量
        vector = list(map(float, vector))  # 变成浮点数
        vocabulary_vectors.append(vector)
        
    vocabulary_vectors = np.array(vocabulary_vectors)

    # 直接tm给你转成哈希表，傻子才用list一个一个搜索呢，堪称头部螺旋桨
    word2index = {} # word->index
    for i in range(len(word_list)):
        word2index[word_list[i]]=i
    
    return word2index, vocabulary_vectors


########################################################################################
# 以下是处理数据的三个函数，分别是：处理一个句子，处理所有文件里的句子，把所有数据打包存起来

SENTENCE_MAXLEN=250
def process_single_sentence(sentence: str, word2index: dict):
    '''
    Process a single sentence to [word_ids]. 
    '''
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    sentence = sentence.replace('\n', ' ').replace('<br /><br />', ' ')
    sentence = re.sub(r, ' \g<0> ', sentence) # 在标点符号左右加空格，为了让标点也独立成为单词
    sentence = sentence.split(' ')
    vec = [sentence[i].lower() for i in range(len(sentence)) if sentence[i] != '']
    
    # 把一个全是小写英文单词的list转换成单词id的list。
    temp = []
    index = -114514
    for j in range(len(vec)):
        try:
            index = word2index[vec[j]]
        except KeyError:  # 没找到
            index = 400000 # 400000 在 glove6B里是 <unk>的 index
        finally:
            temp.append(index)  # index表示一个单词在词典中的id
            
    # 处理成规定长度
    for i in range(len(temp), SENTENCE_MAXLEN):  # 不能补 0 因为 0 是 the 的 index, 这里补 -1 转换成词向量时特殊处理
        temp.append(-1)
    if len(temp) > SENTENCE_MAXLEN:
        temp = temp[0:SENTENCE_MAXLEN]  # 只保留250个
    return temp


def load_data(path, word2index, flag='train'):
    '''
    Open data from files and process all txts into [[[word IDs], label], [[word IDs], label], ...]. 
    '''
    labels = ['pos', 'neg']
    data = []
    for label in labels:
        files = os.listdir(os.path.join(path, flag, label))
        for file in tqdm.tqdm(files): # Don't be nervous, tqdm is only a progress bar. 
            with open(os.path.join(path, flag, label, file), 'r', encoding='utf8') as rf:
                temp = rf.read()
                temp = process_single_sentence(temp, word2index)
                if label == 'pos':
                    data.append([temp, 1])
                elif label == 'neg':
                    data.append([temp, 0])
    return data


def process_sentence(word2index: dict, flag: str, path: str = 'data/aclImdb', length_limit: int = SENTENCE_MAXLEN):
    '''Process data into numpy arrays and save them. 
    ---
    They look like: 
    
    sentence_code: [[word IDs], [word IDs], ...]
    
    labels: [label, label, ...]
    
    flag should be either "train" or "test". 
    '''
    output_dir = os.path.join("./npys", flag)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, "sentence_code.npy")) and os.path.exists(os.path.join(output_dir, "labels.npy")):
        print("大哥，你已经预处理过了🤣不过为了保险起见，还是重新预处理一下")
    
    sentence_code = []
    labels = []
    test_data = load_data(path, word2index, flag)
    
    for i in tqdm.tqdm(range(len(test_data))):
        # nb
        # print(i)
        temp = test_data[i][0]
        label = test_data[i][1] # 0 or 1 0 means neg 1 means positive

        sentence_code.append(temp)
        labels.append(label)
     

    sentence_code = np.array(sentence_code)
    np.save(os.path.join(output_dir, "sentence_code"), sentence_code)
    np.save(os.path.join(output_dir, "labels"), labels)
    
# 数据处理到此完成。
########################################################################################