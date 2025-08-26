import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 将文件路径替换为你的实际 CSV 文件路径
file_path = 'K:\Project\SEED-EEG-Deep-neural-network-main\lightning_logs\SE+transformer/version_143/metrics.csv'

# 1. 定义多个结果文件及对应标签
file_paths = {
    'Fold 1': 'K:\Project\SEED-EEG-Deep-neural-network-main\lightning_logs\SE+transformer/version_131/metrics.csv',
    'Fold 2': 'K:\Project\SEED-EEG-Deep-neural-network-main\lightning_logs\SE+transformer/version_134/metrics.csv',
    'Fold 3': 'K:\Project\SEED-EEG-Deep-neural-network-main\lightning_logs\SE+transformer/version_137/metrics.csv',
    'Fold 4': 'K:\Project\SEED-EEG-Deep-neural-network-main\lightning_logs\SE+transformer/version_140/metrics.csv',
    'Fold 5': 'K:\Project\SEED-EEG-Deep-neural-network-main\lightning_logs\SE+transformer/version_143/metrics.csv',
    # 如有更多文件，可继续添加
}

def plots():
    # 2. 准备颜色列表（数量 ≥ 文件数）
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    # 3. Accuracy 比较图
    plt.figure()
    for (label, fp), color in zip(file_paths.items(), colors):
        df = pd.read_csv(fp)
        # 分离 train/val
        train_df = df[df['train_accuracy'].notna()]
        val_df   = df[df['val_accuracy'].notna()]
        # 画 train：实线
        # plt.plot(train_df['epoch'], train_df['train_accuracy'],
        #          label=f'{label} Train', color=color, linestyle='-')
        # 画 val：虚线
        plt.plot(val_df['epoch'], val_df['val_accuracy'],
                 label=f'{label} Val',   color=color, linestyle='-')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.title('Train vs Validation Accuracy Across Runs')
    plt.legend()
    plt.grid(True)

    # 4. Loss 比较图
    plt.figure()
    for (label, fp), color in zip(file_paths.items(), colors):
        df = pd.read_csv(fp)
        train_df = df[df['train_loss'].notna()]
        val_df   = df[df['val_loss'].notna()]
        # plt.plot(train_df['epoch'], train_df['train_loss'],
        #          label=f'{label} Train', color=color, linestyle='-')
        plt.plot(val_df['epoch'],   val_df['val_loss'],
                 label=f'{label} Val',   color=color, linestyle='-')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Train vs Validation Loss Across Runs')
    plt.legend()
    plt.grid(True)

    # 5. f1 比较图
    plt.figure()
    for (label, fp), color in zip(file_paths.items(), colors):
        df = pd.read_csv(fp)
        # 分离 train/val
        train_df = df[df['train_accuracy'].notna()]
        val_df   = df[df['val_f1score'].notna()]
        # 画 train：实线
        # plt.plot(train_df['epoch'], train_df['train_accuracy'],
        #          label=f'{label} Train', color=color, linestyle='-')
        # 画 val：虚线
        plt.plot(val_df['epoch'], val_df['val_f1score'],
                 label=f'{label} Val',   color=color, linestyle='-')

    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    # plt.title('Train vs Validation Accuracy Across Runs')
    plt.legend()
    plt.grid(True)

    plt.show()



def plot():
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    # 读取并分离训练/验证数据
    train_df = df[df['train_accuracy'].notnull()]
    val_df = df[df['val_accuracy'].notnull()]

    # 绘制准确率曲线
    plt.figure()
    plt.plot(train_df['epoch'], train_df['train_accuracy'], label='Train Accuracy')
    plt.plot(val_df['epoch'], val_df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.title('Train vs Validation Accuracy')
    plt.grid(True)
    plt.legend()

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_df['epoch'], train_df['train_loss'], label='Train Loss')
    plt.plot(val_df['epoch'], val_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Train vs Validation Loss')
    plt.grid(True)
    plt.legend()

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_df['epoch'], train_df['train_f1score'], label='Train F1 Score')
    plt.plot(val_df['epoch'], val_df['val_f1score'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    # plt.title('Train vs Validation Loss')
    plt.grid(True)
    plt.legend()

    plt.show()

def matrix(cm):

    cm_test = np.array([
        [2, 3, 4],  # True negative → pred neg, neu, pos
        [1, 2, 3],  # True neutral  → pred neg, neu, pos
        [1, 3, 5]  # True positive → pred neg, neu, pos
    ])
    disp = ConfusionMatrixDisplay(cm, display_labels=['negative', 'neutral', 'positive'])  # 如果你的标签是 1/2/3
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    # 关闭科学计数法
    # ax = disp.ax_
    # ax.ticklabel_format(style='plain', axis='both')
    plt.title(f"Fold {4} Confusion Matrix")
    plt.show()

