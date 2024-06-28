

## **CMACModel.py：** 

self.hidden_trans中的设计是为了在保留一定程度的空间信息的同时减少特征维度，

这对于某些需要局部特征响应或特征图结构的任务可能有帮助，比如在某些视觉问答、目标检测或图像分割任务中，这样的特征表示可能会更加有效。


## **nn.MultiheadAttention**
重要

https://blog.csdn.net/qq_41813454/article/details/139485125

https://blog.csdn.net/qq_35629563/article/details/130836324
### 参数说明
- embed_dim：表示输入和输出的特征维度。每个注意力头的输出维度是embed_dim除以num_heads，并且在最后会通过线性变换恢复到原始的embed_dim。

- num_heads：指定了并行注意力头的数量。每个头独立地对输入进行注意力计算，然后将所有头的结果拼接起来。

### 输入输出
output, attn_weights = multihead_attn(query, key, value)
- query: 形状为(batch_size, query_sequence_length, embed_dim)的张量，代表查询序列。（即输入序列，这个序列要去注意别的序列）。并且这里的embed_dim需要和MultiheadAttention初始化传入的embed_dim相同。
- key: 形状同样为(batch_size, source_sequence_length, k_dim)的张量，用于与查询序列进行比较以确定注意力分布。（即要去注意哪一个序列）.k_dim, v_dim默认与embed_dim相同。
- value: 形状为(batch_size, source_sequence_length, v_dim)的张量，存储了序列中每个位置的实际值信息，这些值将根据注意力权重进行加权。


- output: 形状为(batch_size, query_sequence_length, embed_dim)的张量（和query相同），这是经过多头注意力机制处理后的序列，每个位置的向量综合考虑了其他位置的信息。
- attn_weights: 形状为(batch_size, num_heads, query_sequence_length, source_sequence_length)的张量，表示在计算注意力时分配给每个位置的权重

## 亮点函数用法
### torch.cat([text_feature, img_text_attention_out], dim=1)
dim=1表示要变化的dim是1.即列的数目要变化，也就是说，待拼接的两个对象的行数要相同，例如(64, 100)&(64,20) -> (64, 120)
再例如torch.cat([text_hidden_state, img_hidden_state], dim=0): (sequence_length, batch, middle_hidden_size)&(img_hidden_seq, batch, middle_hidden_size) -> (sequence_length+img_hidden_seq, batch, middle_hidden_size) 

### attention = TransformerEncoderLayer 和 attention = MultiheadAttention
他们的输入输出用法类似，只不过TransformerEncoderLayer是对输入求自注意力。

    self.attention = nn.TransformerEncoderLayer(
        d_model=config.middle_hidden_size,
        nhead=config.attention_nhead, 
        dropout=config.attention_dropout
    )

    attention_out = self.attention(torch.cat([text_hidden_state, img_hidden_state], dim=0)) 

其中的d_model表示输入特征数。(也就是本项目中的middle_hiddensize)

初始化模型时默认不启用batch_first的时候，输入的数据维度应该为(batch, sequence, in_features)->(sequence, batch, in_features)

输出的形状同样是 [seq_length, batch_size, d_model]

### Train.py中的params
这段代码定义了一个参数分组字典，用于在创建优化器（如AdamW）时，针对模型中特定参数设置独立的学习率和权重衰减策略

            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.config.bert_learning_rate, 'weight_decay': 0.0},
- for n, p in self.model.text_model.bert.named_parameters()：获取BERT模型的所有参数，以(name, parameter)对的形式返回，其中name是参数的名称，parameter是参数本身。
- if any(nd in n for nd in no_decay)：如果参数名n中包含在no_decay列表中的任何一个字符串nd
- 'weight_decay': 0.0：明确指出这一组参数的权重衰减系数为0，意味着这些参数在优化过程中不会应用权重衰减。

## 搬运价值
### 全连接分类器
        # 全连接分类器
        self.text_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )

这一段很重要。根据这一段来对之前写的音频网络输出进行分类，输出分类概率的格式。
