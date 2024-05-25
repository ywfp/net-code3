import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from .fc import FCNet, BCNet
import torch.nn.functional as F

class BaseAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(BaseAttention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class UpDnAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(UpDnAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    # 计算了注意力机制的 logits（对数概率）
    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]、
        # print("v_proj: ")
        # print(v_proj.size()) [128, 36, 1024]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        # print("q_proj: ")
        # print(q_proj.size()) [128, 36, 1024]
        joint_repr = v_proj * q_proj
        # print("joint_repr: ")
        # print(joint_repr.size()) [128, 36, 1024]
        joint_repr = self.dropout(joint_repr)
        # print("joint_repr: ")
        # print(joint_repr.size()) [128, 36, 1024]
        logits = self.linear(joint_repr)
        # print("logits: ")
        # print(logits.size()) [128, 36, 1]
        return logits


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, encoded_text):
        batch_size, seq_length, hidden_size = encoded_text.size()
        
        # Calculate query, key, and value
        query = self.query(encoded_text)  # (batch_size, seq_length, hidden_size)
        key = self.key(encoded_text)  # (batch_size, seq_length, hidden_size)
        value = self.value(encoded_text)  # (batch_size, seq_length, hidden_size)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(1, 2))  # (batch_size, seq_length, seq_length)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_length, seq_length)
        
        # Apply attention weights to value
        attended_text = torch.matmul(attention_weights, value)  # (batch_size, seq_length, hidden_size)
        
        return attended_text


class SanAttention(nn.Module):
  def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
    super(SanAttention, self).__init__()
    # 对图片特征进行卷积 [2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False]
    self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
    # 对问题嵌入向量进行线性变换 [in_features=1024, out_features=512, bias=True]
    self.q_lin = nn.Linear(q_features, mid_features)
    # 对于中间结果进行卷积得到注意力权重 [512, 2, kernel_size=(1, 1), stride=(1, 1)]
    # mid_features为中间卷积层的输出通道数
    # glimpses为注意力头的数量
    self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

    # [p=0.5, inplace=False]
    self.drop = nn.Dropout(drop)
    # ReLU变种：在输入为负数时不输出0而是一个非常小的负数 
    # inplace表示在计算过程中可以修改输入张量，以节省内存空间。
    self.relu = nn.LeakyReLU(inplace=True)

  def forward(self, v, q):
    v = self.v_conv(self.drop(v))
    q = self.q_lin(self.drop(q))
    q = tile_2d_over_nd(q, v)
    x = self.relu(v + q)
    x = self.x_conv(self.drop(x))
    return x

def tile_2d_over_nd(feature_vector, feature_map):
  """ Repeat the same feature vector over all spatial positions of a given feature map.
    The feature vector should have the same batch size and number of features as the feature map.
  """
  n, c = feature_vector.size()         # batchsize和特征数量
  spatial_size = feature_map.dim() - 2 # 空间维度数H、W，即为2
  # 创建一个大小为 (1,1,...,1) 的元组，维数为spatial_size，以用作feature_vector新形状
  # 将feature_vector重塑为大小为 (n,c,1,1) 的 tensor（与feature_map相同）
  # 将 feature_vector 重复到feature_map形状与大小，即每个spatial位置上都有相同的特征向量
  tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
  return tiled

# 将注意力权重矩阵应用到输入矩阵上生成加权矩阵
def apply_attention(input, attention):
  """ Apply any number of attention maps over the input.
    The attention map has to have the same size in all dimensions except dim=1.
  """
  # import pdb
  # pdb.set_trace()
  # input：[batch_size, feature_dim, height, width]
  # attention：[batch_size, glimpses, height, width]
  n, c = input.size()[:2] # [128, 2048]
  glimpses = attention.size(1) # 2

  # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
  # (n, c, h, w) 转换为 (n, c, h*w)
  # view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
  input = input.view(n, c, -1)                # [128,2048, 196]
  attention = attention.view(n, glimpses, -1) # [128, 2, 196]
  s = input.size(2) # 展开后的空间尺寸大小 [196]

  # apply a softmax to each attention map separately  在每个注意力映射上执行 softmax 操作
  # softmax 只能处理二维输入，因此需要将注意力矩阵的前两个维度合并在一起，以便将每个注意力映射分别归一化
  # attention：[batch_size * glimpses, height * width]
  attention = attention.view(n * glimpses, -1) # [256, 196]
  attention = F.softmax(attention)

  # apply the weighting by creating a new dim to tile both tensors over
  target_size = [n, glimpses, c, s]                    # [128, 2, 2048, 196]
  input = input.view(n, 1, c, s).expand(*target_size)  # [128, 2, 2048, 196]
  attention = attention.view(n, glimpses, 1, s).expand(*target_size) # [128, 2, 2048, 196]
  weighted = input * attention # 加权矩阵  [128, 2, 2048, 196]
  # sum over only the spatial dimension
  weighted_mean = weighted.sum(dim=3)  # [128, 2, 2048, 1]
  # the shape at this point is (n, glimpses, c, 1)
  return weighted_mean.view(n, -1)     # [128, 4096]


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2, .5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
                                  name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits
