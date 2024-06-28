import torch
from torch.optim import AdamW
from tqdm import tqdm


class Trainer():
    def __init__(self, config, processor, model,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        # 初始化训练器
        self.config = config  # 训练配置
        self.processor = processor  # 数据预处理和评估工具
        self.model = model.to(device)  # 将模型移到指定设备上
        self.device = device  # 设备（GPU或CPU）

        # 分离BERT和ResNet的参数以应用不同的学习率
        bert_params = set(self.model.text_model.bert.parameters())  # BERT参数集合
        resnet_params = set(self.model.img_model.full_resnet.parameters())  # ResNet参数集合
        other_params = list(set(self.model.parameters()) - bert_params - resnet_params)  # 其他模型参数

        # 定义哪些参数不参与weight decay（通常为偏置项和LayerNorm的权重）
        no_decay = ['bias', 'LayerNorm.weight']

        # 定义优化器参数组
        params = [
            # BERT参数分组，区分是否weight decay
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'lr': self.config.bert_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.config.bert_learning_rate, 'weight_decay': 0.0},

            # ResNet参数分组，同样区分是否weight decay
            {'params': [p for n, p in self.model.img_model.full_resnet.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'lr': self.config.resnet_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.img_model.full_resnet.named_parameters() if
                        any(nd in n for nd in no_decay)],
             'lr': self.config.resnet_learning_rate, 'weight_decay': 0.0},

            # 其他模型参数
            {'params': other_params, 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
        ]

        # 初始化AdamW优化器，根据配置设置学习率和weight decay
        self.optimizer = AdamW(params, lr=config.learning_rate)

    def train(self, train_loader):
        # 训练模型
        self.model.train()  # 设置模型为训练模式

        loss_list = []  # 存储每批的损失
        true_labels, pred_labels = [], []  # 存储真实标签和预测标签用于评估

        # 遍历训练数据加载器
        for batch in tqdm(train_loader, desc='----- [Training] '):
            guids, texts, texts_mask, imgs, labels = batch  # 解包批次数据，guids是每个样本的唯一标识符
            texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(
                self.device), labels.to(self.device)  # 移动数据至设备

            # 前向传播并计算损失
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels)

            # 收集损失和标签用于后续评估
            loss_list.append(loss.item())
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

            # 反向传播、梯度清零、更新参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 计算所有batch的平均训练损失
        train_loss = round(sum(loss_list) / len(loss_list), 4)
        return train_loss, loss_list

    def valid(self, val_loader):
        # 验证模型
        self.model.eval()  # 设置模型为评估模式

        val_loss = 0  # 初始化验证损失
        true_labels, pred_labels = [], []  # 初始化标签列表

        # 遍历验证数据加载器
        for batch in tqdm(val_loader, desc='\t ----- [Validing] '):
            guids, texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(
                self.device), labels.to(self.device)

            # 前向传播并收集损失
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels)
            val_loss += loss.item()

            # 收集标签用于评估
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

        # 计算平均验证损失并进行评估
        avg_val_loss = val_loss / len(val_loader)
        metrics = self.processor.metric(true_labels, pred_labels)  # 使用处理器计算评估指标
        return avg_val_loss, metrics

    def predict(self, test_loader):
        # 预测
        self.model.eval()  # 设置模型为评估模式

        pred_guids, pred_labels = [], []  # 初始化预测结果容器

        # 遍历测试数据加载器
        for batch in tqdm(test_loader, desc='----- [Predicting] '):
            guids, texts, texts_mask, imgs, _ = batch
            texts, texts_mask, imgs = texts.to(self.device), texts_mask.to(self.device), imgs.to(self.device)

            # 前向传播获取预测结果
            pred = self.model(texts, texts_mask, imgs)

            # 收集预测结果
            pred_guids.extend(guids)
            pred_labels.extend(pred.tolist())

        # 返回预测结果的配对（guid, 预测标签）
        return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]