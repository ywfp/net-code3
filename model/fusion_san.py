import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import freeze_layer
from torch.autograd import Variable
from .attention import SanAttention, apply_attention
from .fc import GroupMLP
from .language_model import Seq2SeqRNN, WordEmbedding
import pdb

class SAN(nn.Module):
    #args, self.train_loader.dataset, self.question_word2vec
    #def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args, dataset,embedding_weights=None,rnn_bidirectional=True):
        super(SAN, self).__init__()
        embedding_requires_grad = not args.freeze_w2v # 是否在训练模型时更新embedding权重
        question_features = 1024
        # RNN的特征维度，如果是双向RNN则除以2
        rnn_features = int(question_features // 2) if rnn_bidirectional else int(question_features)
        vision_features = args.output_features  # self.output_features = 2048
        # glimpses = 2 # SAN网络中注意力头的数量
        glimpses = 2

        # vocab_size = embedding_weights.size(0)
        # vector_dim = embedding_weights.size(1)
        # self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        # self.embedding.weight.data = embedding_weights
        # self.embedding.weight.requires_grad = embedding_requires_grad
        
        # 将词汇表中的单词转换为预训练的词向量表示 15421
        self.w_emb = WordEmbedding(embedding_weights.size(0), 300, .0)
        # raise Exception(embedding_weights.size(0))
        # 训练过程中w_emb不更新权重
        if args.freeze_w2v:
            self.w_emb.init_embedding(embedding_weights)
            freeze_layer(self.w_emb)

        self.drop = nn.Dropout(0.5)
        # 将输入的词嵌入向量映射到低维空间，然后通过双向LSTM网络对输入进行编码，以学习单词之间的上下文信息
        # [300, 512, bidirectional=True]
        self.text = Seq2SeqRNN(
            input_features=embedding_weights.size(1),
            rnn_features=int(rnn_features),
            rnn_type='LSTM',
            # rnn_type='GRU',
            rnn_bidirectional=rnn_bidirectional, # rnn_bidirectional=True则使用双向LSTM编码
        )
        # [2048, 1024, 512, 2, 0.5]
        self.attention = SanAttention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            # glimpses=2,
            glimpses=2,
            drop=0.5,
        )
        # [5120, 8192, 1024]
        self.mlp = GroupMLP(
            in_features=glimpses * vision_features + question_features,
            mid_features= 4 * args.hidden_size,  # 4*2048=8192
            out_features=args.embedding_size,    # 1024
            drop=0.5,
            groups=64,
        )

        # 对神经网络的参数（如权重和偏差）进行初始化，以帮助网络更快地学习数据的特征
        for m in self.modules():
            # 某个模块是线性层或二维卷积层，则对其权重进行Xavier初始化
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self, v, b, q, q_len):
        # pdb.set_trace()
        # q是glove编码，GloVe通过学习全局统计信息来为单词生成固定维度的词向量表示
        # GRU则是一种序列模型，用于捕捉序列数据中的上下文信息
        # print(q.shape) [128, 25]
        # print((self.w_emb(q)).shape) [128, 25, 300]
        # print((self.drop(self.w_emb(q))).shape) [128, 25, 300]
        
        # print(q_len.shape)
        # print((q_len.data).shape)
        # raise Exception((list(q_len.data)).shape)
        
        # [128, 1024]
        q = self.text(self.drop(self.w_emb(q)), list(q_len.data))
        # q = self.text(self.embedding(q), list(q_len.data))
        # raise Exception(q.shape)

        # print("q: ")
        # print(q.size()) 
        
        # [128, 2048, 14, 14]
        v = F.normalize(v, p=2, dim=1)
        # print("v: ")
        # print(v.size()) 
        
        # [128, 2, 14, 14]
        a = self.attention(v, q)  # 权重
        # print("a: ")
        # print(a.size()) 
        
        # [128, 4096]
        v = apply_attention(v, a) # 加权特征向量 [n, glimpses * c]
        # print("v: ")
        # print(v.size()) 

        # [128, 5120]
        combined = torch.cat([v, q], dim=1)
        # print("combined: ")
        # print(combined.size()) 
        
        # [128, 1024]
        embedding = self.mlp(combined)
        # print("embedding: ")
        # print(embedding.size()) 
        return embedding