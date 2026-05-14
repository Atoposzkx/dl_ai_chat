
#1、导入第三方库
# import numpy as np
import pickle

#2、创建类
class word_ip_map(object):
    idx2w=[]
    w2idx=[]
    def __init__(self):
        with open('./data/idx2w.pkl','rb') as f:
            self.idx2w=pickle.load(f)
        with open('./data/w2idx.pkl','rb') as f:
            self.w2idx=pickle.load(f)
    def sentence2ids(self,sentence):
        ids=[]
        for word in sentence:
            ids.append(self.w2idx.get(word,self.w2idx.get('unk')))
        return ids
    def ids2sentence(self,ids):
        sentence=[]
        for id in ids:
            sentence.append(self.idx2w[id])
        return sentence

#3、定义主函数
def main():
    map=word_ip_map()
    ids=map.sentence2ids(['hello','world','are','you','ok','i','am','ok','你','好','中国'])
    print('input_ids',ids)
    sentence=map.ids2sentence(ids)
    print('input_sentence',sentence)

#4、运行主函数
if __name__=='__main__':
    main()

