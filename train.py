
# 1、导入第三方库
import tensorflow as tf
# from Map_Word_Id import word_ip_map
from seq2seq_model import Seq_SeqModel

#2、定义超参数
batch_size=27
sequence_length=50
hidden_size=256
num_layers=2
num_encoder_symbols=5004
num_decoder_symbols=5004
embedding_size=256
learning_rate=0.001

#3、定义主函数，实例化模型并进行训练
def main():
    epochs=1000
    graph=tf.Graph()
    with graph.as_default():
        model=Seq_SeqModel(hidden_size,num_layers,batch_size,sequence_length,\
                           embedding_size,learning_rate,num_encoder_symbols,num_decoder_symbols,'false')
        with tf.Session(graph=graph) as session:
            model.train(session,epochs)
#4、执行主函数
if __name__=='__main__':
    main()