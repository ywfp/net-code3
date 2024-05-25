import torch
import torch.nn as nn
from .language_model import WordEmbedding
from utils import freeze_layer

# 定义 LSTM 编码器
class LSTMEncoder(nn.Module):
    def __init__(self, args, embedding_weights=None):
        super(LSTMEncoder, self).__init__()
        # 将词汇表中的单词转换为预训练的词向量表示 15421
        self.w_emb = WordEmbedding(embedding_weights.size(0), 300, .0)
        # raise Exception(embedding_weights.size(0))
        # 训练过程中w_emb不更新权重
        if args.freeze_w2v:
            self.w_emb.init_embedding(embedding_weights)
            freeze_layer(self.w_emb)
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(300, 512, 1, bidirectional=True)
        self.attention = nn.Linear(1024, 1)

    def forward(self, embeddings):
        # 输入的q为词嵌入向量，形状为 [batch_size, seq_len, input_size]
        # 输出的outputs为LSTM编码后的输出，形状为 [batch_size, seq_len, hidden_size * num_directions]
        # 输出的hidden为LSTM最后一个时间步的隐状态，形状为 [num_layers * num_directions, batch_size, hidden_size]
        # 输出的cell为LSTM最后一个时间步的细胞状态，形状为 [num_layers * num_directions, batch_size, hidden_size]
        q = self.drop(self.w_emb(embeddings))
        outputs, (hidden, cell) = self.lstm(q)
        # 输入的inputs为LSTM编码后的输出，形状为 [batch_size, seq_len, hidden_size]
        # 计算注意力权重
        attention_weights = self.attention(outputs).squeeze(-1)  # 形状为 [batch_size, seq_len]
        attention_weights = torch.softmax(attention_weights, dim=1)  # 形状为 [batch_size, seq_len]
        # 使用注意力权重对编码后的输出进行加权求和
        out = torch.matmul(attention_weights.unsqueeze(1), outputs).squeeze(1)  # 形状为 [batch_size, hidden_size]
        return out