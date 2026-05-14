
# 1、导入第三方库
import nltk, itertools, pickle, numpy as np
import jieba,tensorflow as tf

# 2初始化变量
FILENAME = 'wenda.txt'
# FILENAME = 'test.txt'
limit = {'maxq': 50, 'minq': 0, 'maxa': 48, 'mina': 3}
UKN = 'unk'
GO = '<go>'
EOS = '<eos>'
PAD = '<pad>'
VOCAB_SIZE = 5000


# 3、定义子函数
def cut_word(sentence):
    seq_list=jieba.cut(sentence)
    return tf.compat.as_str("/".join(seq_list)).split('/')


def index_(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    # print('vocab : ',vocab)
    index2word = [GO] + [EOS] + [UKN] + [PAD] + [x[0] for x in vocab]
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dist


def zero_pad(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)
    idx_o = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)
    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'], 1)
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 2)
        o_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 3)
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
        idx_o[i] = np.array(o_indices)
    return idx_q, idx_a, idx_o


def pad_seq(seq, lookup, maxlen, flag):
    if flag == 1:
        indices = []
    if flag == 2:
        indices = [lookup[GO]]
    if flag == 3:
        indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UKN])
    if flag == 1:
        return indices + [lookup[PAD]] * (maxlen - len(seq))
    if flag == 2:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq))
    if flag == 3:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq) + 1)

#4、数据预处理
def process_data():
    qtokenized,atokenized=[],[]
    lines=open(FILENAME,encoding='UTF-8').read().split('\n')
    # print('lines : ',len(lines),lines)
    data_len=len(lines)
    for i in range(0,data_len,2):
        qline,aline=lines[i],lines[i+1]
        # print('qline : ',qline)
        qline,aline=qline.lower(),aline.lower()
        qlist=qline.split('+++$+++')
        alist=aline.split('+++$+++')
        # print('qlist : ',qlist)
        qline=qlist[-1][1:]
        aline = qlist[-1][1:]
        # print('qline : ',qline)
        qWords,aWords=cut_word(qline),cut_word(aline)
        # print('qWords : ',qWords)
        if limit['maxq'] >= len(qWords) and limit['minq'] <= len(qWords) and \
                limit['maxa'] >= len(aWords) and limit['mina'] <= len(aWords):
            qtokenized.append(qWords)
            atokenized.append(aWords)
    filter_len=len(qtokenized)
    filtered=100-int(filter_len*100/(data_len//2))
    print(str(filtered)+"% 的数据被过滤")
    idx2w, w2idx, freq_dist=index_(qtokenized+atokenized, vocab_size=VOCAB_SIZE)
    print('idx2w len: {0} ,idx2w {1}'.format(len(idx2w),idx2w))
    print('w2idx len: {0} ,w2idx {1}'.format(len(w2idx), w2idx))
    with open('./data/idx2w.pkl','wb') as f:
        pickle.dump(idx2w,f)
    with open('./data/w2idx.pkl','wb') as f:
        pickle.dump(w2idx,f)
    idx_q, idx_a, idx_o=zero_pad(qtokenized, atokenized, w2idx)
    np.save('./data/idx_q.npy',idx_q)
    np.save('./data/idx_a.npy',idx_a)
    np.save('./data/idx_o.npy', idx_o)
#5、主程序入口
if __name__=='__main__':
    process_data()