import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50


class TextModel(nn.Module):

    def __init__(self, config):
        # 初始化方法接收一个配置对象config，该对象包含了模型构建所需的各种参数和设置。
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)         # 加载预训练的BERT模型
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),         # 将BERT的输出映射到Fuse中间层的隐藏尺寸
            nn.ReLU(inplace=True)                                                       # self.bert.config.hidden_size：BERT模型中Transformer编码器的隐藏层维度
        ) 
        
        # 是否进行fine-tune--BERT模型的参数是否参与梯度更新
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False         # 结BERT模型的参数，只训练后续添加的层
            else:
                param.requires_grad = True
        
        # self.bert.init_weights()

    def forward(self, bert_inputs, masks, token_type_ids=None):
        '''
        bert_inputs(输入的词ID); masks(注意力掩码); token_type_ids(用于区分句子对中的两个句子)。
        '''
        assert bert_inputs.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        pooler_out = bert_out['pooler_output']          # 取其pooler_output，这是BERT模型顶层的池化输出，常用于后续的分类任务。
        
        return self.trans(pooler_out)


class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)        # 使用torchvision内置的resnet50模型，并加载预训练权重。

        self.resnet = nn.Sequential(                            # self.full_resnet.children()：这个方法会返回模型(self.full_resnet)的所有子模块
            *(list(self.full_resnet.children())[:-1]),          # [:-1]切片操作，从列表的开始一直取到倒数第二个元素, 这会排除最后一个全连接fc层（默认的分类头）
            nn.Flatten()                                        # Flatten展平为二维张量（(batch_size, channels * height * width)）
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),      # self.full_resnet.fc.in_features: 原ResNet模型中全连接fc层所期待的输入特征维度
            nn.ReLU(inplace=True)
        )
        
        # 是否进行fine-tune
        for param in self.full_resnet.parameters():
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, imgs):
        feature = self.resnet(imgs)

        return self.trans(feature)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # 上面的TextModel
        self.text_model = TextModel(config)
        # 上面的ImageModel
        self.img_model = ImageModel(config)

        # 全连接分类器
        self.text_classifier = nn.Sequential(                               # 定义了文本特征到分类结果的全连接网络
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)                                               # 最后使用Softmax函数使输出成为合法的概率分布。
        )
        self.img_classifier = nn.Sequential(                                # 定义了图像特征到分类结果的全连接网络
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor(config.loss_weight))       # 损失函数：可以根据配置文件中的loss_weight来调整不同类别的损失权重，这对于类别不平衡的数据集尤为重要。

    def forward(self, texts, texts_mask, imgs, labels=None):
        text_feature = self.text_model(texts, texts_mask)                   # 使用TextModel和ImageModel特征提取
        img_feature = self.img_model(imgs)

        text_prob_vec = self.text_classifier(text_feature)                  # text, image分别通过各自的分类器得到概率向量
        img_prob_vec = self.img_classifier(img_feature)
        prob_vec = torch.softmax((text_prob_vec + img_prob_vec), dim=1)     # 简单地将这两个概率向量相加以融合两者的信息,接着通过Softmax函数得到最终的分类概率分布
        pred_labels = torch.argmax(prob_vec, dim=1)                         # argmax找到最可能的类别作为预测标签

        if labels is not None:
            loss = self.loss_func(prob_vec, labels)     # 计算预测概率分布与真实标签之间的交叉熵损失
            return pred_labels, loss                    # 返回预测标签和损失值
        else:
            return pred_labels
