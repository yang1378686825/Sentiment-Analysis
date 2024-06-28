import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁止tokenizer并行化以避免潜在冲突
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息以保持输出清晰
import sys
sys.path.append('./utils')  # 添加目录到系统路径以便导入自定义模块
sys.path.append('./utils/APIs')

import torch

import argparse
from Config import config
from utils.common import data_format, read_from_file, train_val_split, save_model, write_to_file
from utils.DataProcess import Processor
from Trainer import Trainer


# args
parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true', help='训练模型')
parser.add_argument('--text_pretrained_model', default='roberta-base', help='文本分析模型', type=str)
parser.add_argument('--fuse_model_type', default='OTE', help='融合模型类别', type=str)
parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
parser.add_argument('--weight_decay', default=1e-2, help='设置权重衰减', type=float)
parser.add_argument('--epoch', default=5, help='设置训练轮数', type=int)

parser.add_argument('--do_test', action='store_true', help='预测测试集数据')
parser.add_argument('--load_model_path', default=None, help='已经训练好的模型路径', type=str)
parser.add_argument('--text_only', action='store_true', help='仅用文本预测')
parser.add_argument('--img_only', action='store_true', help='仅用图像预测')

args = parser.parse_args()      # 解析命令行参数，下方根据命令行参数更新config配置
config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
config.bert_name = args.text_pretrained_model
config.fuse_model_type = args.fuse_model_type
config.load_model_path = args.load_model_path
config.only = 'img' if args.img_only else None
config.only = 'text' if args.text_only else None
if args.img_only and args.text_only: config.only = None
print('TextModel: {}, ImageModel: {}, FuseModel: {}'.format(config.bert_name, 'ResNet50', config.fuse_model_type))


# Initilaztion
processor = Processor(config)
if config.fuse_model_type == 'CMAC' or config.fuse_model_type == 'CrossModalityAttentionCombine':
    from Models.CMACModel import Model
elif config.fuse_model_type == 'HSTEC' or config.fuse_model_type =='HiddenStateTransformerEncoder':
    from Models.HSTECModel import Model
elif config.fuse_model_type == 'OTE' or config.fuse_model_type == 'OutputTransformerEncoder':
    from Models.OTEModel import Model
elif config.fuse_model_type == 'NaiveCat':
    from Models.NaiveCatModel import Model
else:
    from Models.NaiveCombineModel import Model
model = Model(config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
trainer = Trainer(config, processor, model, device)


# Train
def train():
    # 数据格式化
    # 对训练数据进行预处理,接收三个参数：训练数据的原始文本文件路径、数据存放目录以及要生成的JSON文件路径。
    data_format(os.path.join(config.root_path, './data/train.txt'), 
    os.path.join(config.root_path, './data/data'), os.path.join(config.root_path, './data/train.json'))

    # 读取数据
    # config.train_data_path='data/train.json'、config.data_dir='./data/data/'、config.only指示是否仅使用文本或图像数据。
    data = read_from_file(config.train_data_path, config.data_dir, config.only)

    train_data, val_data = train_val_split(data)                    # 划分为训练集train_data和验证集val_data
    train_loader = processor(train_data, config.train_params)       # 数据加载器dataloader创建
    val_loader = processor(val_data, config.val_params)

    best_acc = 0
    epoch = config.epoch
    for e in range(epoch):
        print('-' * 20 + ' ' + 'Epoch ' + str(e+1) + ' ' + '-' * 20)
        tloss, tloss_list = trainer.train(train_loader)                         # trainer.train训练
        print('Train Loss: {}'.format(tloss))
        vloss, vacc = trainer.valid(val_loader)                                 # trainer.valid验证
        print('Valid Loss: {}'.format(vloss))
        print('Valid Acc: {}'.format(vacc))
        if vacc > best_acc:
            best_acc = vacc
            save_model(config.output_path, config.fuse_model_type, model)       # 模型保存
            print('Update best model!')
        print()                                                                 # 换行打印，在每个epoch结束后打印一个空行


# Test
def test():
    data_format(os.path.join(config.root_path, './data/test_without_label.txt'), 
    os.path.join(config.root_path, './data/data'), os.path.join(config.root_path, './data/test.json'))      # 数据格式化
    test_data = read_from_file(config.test_data_path, config.data_dir, config.only)                         # 读取测试数据
    test_loader = processor(test_data, config.test_params)                                                  # 创建测试数据加载器

    if config.load_model_path is not None:
        model.load_state_dict(torch.load(config.load_model_path))           # 加载模型权重

    outputs = trainer.predict(test_loader)                                  # 模型预测
    formated_outputs = processor.decode(outputs)                            # 解码预测输出
    write_to_file(config.output_test_path, formated_outputs)


# main
if __name__ == "__main__":
    if args.do_train:
        train()
    
    if args.do_test:
        if args.load_model_path is None and not args.do_train:
            print('请输入已训练好模型的路径load_model_path或者选择添加do_train arg')
        else:
            test()