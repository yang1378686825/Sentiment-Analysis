import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50

class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),     # 将BERT的输出映射到Fuse中间层的隐藏尺寸
            nn.ReLU(inplace=True)
        ) 
        
        # 是否进行fine-tune
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # self.bert.init_weights()

    def forward(self, bert_inputs, masks, token_type_ids=None):
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        hidden_state = bert_out['last_hidden_state']        # 整个序列的隐藏状态,形状为(batch_size, sequence_length, hidden_size),保留了整个序列的上下文信息。
        pooler_out = bert_out['pooler_output']              # 句级别的输出，形状为(batch_size, hidden_size),通常用于像情感分析这样的句子级别任务。
        
        return self.trans(hidden_state), self.trans(pooler_out)     # 双路径输出: (batch_size, sequence_length(bert模型自己的参数), middle_hidden_size=64)、（batch_size,middle_hidden_size=64）


class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        self.resnet_h = nn.Sequential(
            *(list(self.full_resnet.children())[:-2]),      # 保留了第一到倒数第三层，去除最后的平均池化层和全连接层
        )

        self.resnet_p = nn.Sequential(
            list(self.full_resnet.children())[-2],          # 最后一个平均池化层（AdaptiveAvgPool2d，默认输出形状为(batch_size, channels, 1, 1)）+ Flatten层
            nn.Flatten()
        )

        # 特殊层
        # (batch,2048,7,7)->（batch,img_hidden_seq,7,7）->(batch,img_hidden_seq,7*7)->(batch, img_hidden_seq, middle_hidden_size)
        self.hidden_trans = nn.Sequential(
            nn.Conv2d(self.full_resnet.fc.in_features, config.img_hidden_seq, 1),       # 通过1x1卷积减少特征通道数: 最后一个卷积层输出的特征通道数(full_resnet.fc.in_features)→配置中设定的新的特征通道数(img_hidden_seq)
            nn.Flatten(start_dim=2),                        # 从第2维度开始拉平，保留了batch维和channel维，只将高度和宽度维度展平
            nn.Dropout(config.resnet_dropout),
            nn.Linear(7 * 7, config.middle_hidden_size),    # 这里的7*7是根据resnet50，原img大小为224*224多次下采样得来的
            nn.ReLU(inplace=True)
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # 是否进行fine-tune
        for param in self.full_resnet.parameters():
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, imgs):
        hidden_state = self.resnet_h(imgs)      # self.resnet_h处理输入图像，得到特征图(batch, 2048, 7, 7)
        feature = self.resnet_p(hidden_state)   # resnet_p+hidden_trans继续resnet网络，得到一个针对整个图像的特征向量，适合于图像级别的分类任务：（batch，middle_hidden_size）
                                                # hidden_trans获得保留一定空间信息的特征向量：(batch, img_hidden_seq=64, middle_hidden_size=64)
        return self.hidden_trans(hidden_state), self.trans(feature)     # 双路径输出↑


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # 上面的TextModel
        self.text_model = TextModel(config)
        # 上面的ImageModel
        self.img_model = ImageModel(config)
        # attention：文本到图像的注意力(text_img_attention)、图像到文本的注意力(img_text_attention)
        self.text_img_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead, 
            dropout=config.attention_dropout,
        )
        self.img_text_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead, 
            dropout=config.attention_dropout
        )

        # 全连接分类器
        self.text_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.img_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, imgs, labels=None):

        text_hidden_state, text_feature = self.text_model(texts, texts_mask)
        img_hidden_state, img_feature = self.img_model(imgs)

        text_hidden_state = text_hidden_state.permute(1, 0, 2)      # 为了与多头注意力机制兼容，需要调整张量的维度，将序列维度（通常是0）移动到最后一位，因此使用permute(1, 0, 2)
        img_hidden_state = img_hidden_state.permute(1, 0, 2)

        text_img_attention_out, _ = self.img_text_attention(img_hidden_state,
                                                            text_hidden_state, text_hidden_state)   # img对text的注意力的输出序列
        text_img_attention_out = torch.mean(text_img_attention_out, dim=0).squeeze(0)               # 之前.permute(1, 0, 2) 了，所以dim=0会将sequence维度压缩为1，.squeeze(0) 会去掉=1的维度：
                                                                                                    # （batch,sequence,middle_hiddensize=64）-> （batch,middle_hiddensize=64）
        img_text_attention_out, _ = self.text_img_attention(text_hidden_state,
                                                            img_hidden_state, img_hidden_state)     # text对img的注意力的输出序列
        img_text_attention_out = torch.mean(img_text_attention_out, dim=0).squeeze(0)

        text_prob_vec = self.text_classifier(torch.cat([text_feature, img_text_attention_out], dim=1))      # 将原始的特征与跨模态注意力得到的特征拼接-> （batch, 2*middle_hiddensize）
        img_prob_vec = self.img_classifier(torch.cat([img_feature, text_img_attention_out], dim=1))         # 通过对应的分类器（text_classifier或img_classifier）得到分类概率向量。

        prob_vec = torch.softmax((text_prob_vec + img_prob_vec), dim=1)     # 将两个模态的预测概率向量相加，得到最终的分类概率prob_vec
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels