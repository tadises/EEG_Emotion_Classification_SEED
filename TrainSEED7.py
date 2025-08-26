import os
import random
import numpy as np
import torch
import gc
import lmdb
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from torcheeg.trainers import ClassifierTrainer
from modeltest import SEED_DEEP
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _prepare_npz_entry(args):
    """
    用于多进程读取并序列化单个 npz 文件，返回 (key, serialized_value).
    args: (idx, fname, npz_dir)
    """
    idx, fname, npz_dir = args
    path = os.path.join(npz_dir, fname)
    data = np.load(path)
    arr = data['data']
    label = int(data['label'])
    serialized = pickle.dumps((arr, label), protocol=pickle.HIGHEST_PROTOCOL)
    key = f"{idx:08d}".encode('ascii')
    return key, serialized


def build_lmdb_from_npz(npz_dir, lmdb_path, map_size=1024**4, batch_size=10000):
    """
    分批、多进程构建 LMDB 数据库。
    npz_dir: npz 文件目录
    lmdb_path: LMDB 存储目录
    map_size: 最大磁盘映射大小
    batch_size: 每个事务提交的条目数
    """
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)

    # 准备所有文件列表
    fnames = [f for f in sorted(os.listdir(npz_dir)) if f.endswith('.npz')]
    total = len(fnames)
    args_list = [(i, fnames[i], npz_dir) for i in range(total)]

    # 使用多进程并行序列化
    workers = max(1, cpu_count() - 1)
    with Pool(workers) as pool:
        it = pool.imap(_prepare_npz_entry, args_list, chunksize=workers)

        txn = env.begin(write=True)
        count = 0
        for key, serialized in it:
            txn.put(key, serialized)
            count += 1
            if count % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()

    env.sync()
    env.close()

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None
        # 只打开临时只读环境来获取总条目数
        tmp_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with tmp_env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        tmp_env.close()

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_env()
        key = f"{idx:08d}".encode('ascii')
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
        arr, label = pickle.loads(data)
        x = torch.tensor(arr, dtype=torch.float32)
        y = label
        return x, y


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)

    npz_dir = r'K:\Download\SEED-VII\SEED-VII\epochs_data\npz'
    lmdb_path = r'K:\Download\SEED-VII\SEED-VII\epochs_data\lmdb_db'

    # 构建 LMDB
    if not os.path.exists(lmdb_path) or not os.listdir(lmdb_path):
        print("Building LMDB database from NPZ with multiprocessing...")
        build_lmdb_from_npz(npz_dir, lmdb_path, map_size=1024**4, batch_size=10000)

    # 数据加载
    dataset = LMDBDataset(lmdb_path)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    bs = 256
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    model = SEED_DEEP(
        do_pool=True, dropoutc=0.2,
        in_channels=1, num_classes=7,
        out_channels=50, num_res_layers=5,
        ll1=128, ll2=768,
        dropout1=0.5, dropout2=0.5
    ).to(device)

    trainer = ClassifierTrainer(
        model=model,
        num_classes=7,
        lr=1e-4,
        weight_decay=1e-4,
        devices=1,
        accelerator='gpu',
        metrics=['accuracy', 'f1score'],
        verbose=True
    )

    trainer.fit(train_loader, val_loader, max_epochs=90)
    trainer.test(val_loader)

    logits_list = trainer.predict(val_loader)
    logits = torch.cat([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in logits_list], dim=0)
    y_pred = logits.argmax(dim=1).cpu().numpy()

    y_true_list = []
    for _, y in val_loader:
        y_true_list.append(y.numpy())
    y_true = np.concatenate(y_true_list, axis=0)

    cm = confusion_matrix(y_true, y_pred)
    labels = ['Disgust','Fear','Sad','Neutral','Happy','Anger','Surprise']
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
