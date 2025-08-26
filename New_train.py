import torch, gc
import os
import random
import numpy as np
# print(torch.__version__)

import logging, time

from mne.simulation.metrics import f1_score


# 1. 固定随机种子
# from transformers import set_seed
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def add_one(x):
    return x + 1



# 2. 导入新版 Trainer
from torcheeg.trainers import ClassifierTrainer  # :contentReference[oaicite:5]{index=5}
from torchmetrics.classification import MulticlassConfusionMatrix


# 3. 其余模块
from torch.utils.data import DataLoader
from torcheeg import transforms
from torcheeg.datasets import SEEDDataset
from torcheeg.model_selection import KFoldGroupbyTrial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

from modeltest import SEED_DEEP

plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_locale'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False

    # 4. 设备和日志
def main():
    set_seed(42)  # 同时设置 random, numpy, torch, tf 等种子 :contentReference[oaicite:4]{index=4}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("GPU: running on", device)
    print("This is a minimal script to recreate best results")

    logger = logging.getLogger('seed-deep-cnn-lstm')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    # 超参数
    sample = 200
    bs = 256
    do_pool = True
    dropout1, dropout2, dropoutc = 0.5, 0.5, 0.2
    num_res_layers = 5
    c1 = 50
    ll1, ll2 = 1024, 768
    epochs = 90
    lr = 1e-4
    wd = 1e-4
    k = 5

    # 配置 torcheeg 这个 logger
    logger = logging.getLogger("torcheeg")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 5. 构建数据集
    dataset = SEEDDataset(
        root_path=r'K:\Download\SEED\Preprocessed_EEG',  # EEG 数据根目录 :contentReference[oaicite:6]{index=6}
        io_path='./seed_full_200/seed',                  # IO 缓存目录
        io_mode='lmdb',
        # io_size=1000 * 1024 * 1024,                     # 1GiB
        chunk_size=sample,                               # 每个样本的时间点数量
        overlap=0,
        num_channel=62,
        offline_transform=transforms.Compose([]),
        online_transform=transforms.Compose([transforms.ToTensor()]),
        label_transform=transforms.Compose([
            transforms.Select('emotion'),
            transforms.Lambda(add_one)
        ]),
        num_worker=4,
        verbose=True

    )

    # 6. K 折划分
    k_fold = KFoldGroupbyTrial(
        n_splits=k,
        split_path='./seed_full_200/split',
        shuffle=True,
        random_state=42
    )  # :contentReference[oaicite:7]{index=7}

    # 7. 训练与测试循环
    for fold_idx, (train_ds, val_ds) in enumerate(k_fold.split(dataset)):
        # 每折重设种子
        set_seed(42)

        # 构建模型（用户自定义）
        # model = SEED_DEEP(
        #     do_pool=do_pool, dropoutc=dropoutc,
        #     in_channels=1, num_classes=3,
        #     out_channels=c1, num_res_layers=num_res_layers,
        #     ll1=ll1, ll2=ll2,
        #     dropout1=dropout1, dropout2=dropout2
        # ).to(device)

        model_test = SEED_DEEP(
            do_pool=do_pool, dropoutc=dropoutc,
            in_channels=1, num_classes=3,
            out_channels=c1, num_res_layers=num_res_layers,
            ll1=128, ll2=ll2,
            dropout1=dropout1, dropout2=dropout2
        ).to(device)

        # print(model)




        # 构建新版 Trainer
        trainer = ClassifierTrainer(
            model=model_test,
            num_classes=3,
            lr=lr,
            weight_decay=wd,
            devices=1,
            accelerator='gpu',
            metrics=[
                'accuracy', "f1score"
            ],
            verbose=True
        )
        # "accuracy"：整体分类准确率（Accuracy）
        # "precision"：宏平均精确率（Precision）
        # "recall"：宏平均召回率（Recall）
        # "f1score"：宏平均
        # F1分数（F1 - Score）
        # "matthews"：MCC（Matthew’s Correlation Coefficient）
        # "auroc"：多类
        # AUROC（Area Under the ROC Curve）
        # "kappa"：Cohen’s Kappa
        # 统计量
        # trainer.logger = logger

        # DataLoader
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=4, pin_memory=False, prefetch_factor=1, persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False,
            num_workers=4, pin_memory=False, prefetch_factor=1, persistent_workers=True
        )

        default_dir = f"lightning_logs/fold_{fold_idx}"
        # 训练 & 测试
        trainer.fit(
            train_loader,
            val_loader,
            max_epochs=epochs,
            # default_root_dir=default_dir,  # 每折一个根目录
            # enable_checkpointing=False,  # 关闭自动保存/加载 checkpoint
        )
        trainer.test(val_loader)

        all_logits = trainer.predict(val_loader)

        # 3. 拼接并取 argmax 得到 y_pred
        #    可能 all_logits 是 Tensor 或 np.array，以下以 Tensor 为例：
        logits = torch.cat(all_logits, dim=0)  # [N, num_classes]
        y_pred = logits.argmax(dim=1).cpu().numpy()  # [N]

        # 4. 收集 y_true
        y_true_list = []
        for _, y in val_loader:
            y_true_list.append(y.numpy())
        y_true = np.concatenate(y_true_list, axis=0)  # [N]

        # 5. 计算 & 可视化混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['negative', 'neutral', 'positive'])  # 如果你的标签是 1/2/3
        disp.plot(cmap=plt.cm.Blues, values_format='d')

        plt.title(f"Fold {fold_idx} Confusion Matrix")
        plt.show()

        torch.cuda.empty_cache()
        gc.collect()






if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()