import torch
import torch.nn as nn


# if depth or size changes and we need to resample for the residual connection addition.
class Resample_Channels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resample_Channels, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layer(x)


# a single residual block with 2 cnn layers.
class SEED_DEEP_LAYER(nn.Module):
    def __init__(self, in_channels, out_channels, in_d_0=0, in_d_1=0, stride=1, k=4, do_pool=False, dropout=0.01,
                 debug=False):
        super(SEED_DEEP_LAYER, self).__init__()
        self.do_pool = do_pool
        self.debug = debug
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())

        conv2_layers = [
            nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=stride, padding='same'),
            nn.BatchNorm2d(out_channels)
        ]
        self.mp = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout2d(dropout)
        self.downsample = nn.AvgPool2d((2, 2))
        # a way to build layers out of a list
        self.conv2 = nn.Sequential(*conv2_layers)
        self.se = SELayer(out_channels, reduction=16)
        # do we need to do downsampling?
        self.resample_channels = False
        if in_channels != out_channels:
            self.resampler = Resample_Channels(in_channels, out_channels)
            self.resample_channels = True
        self.a = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        o = self.conv1(x)
        o = self.dropout(o)
        o = self.conv2(o)
        o = self.se(o)
        # going from in channels to out channels requires the residual to be resampled. so that it can be added
        if self.resample_channels:
            residual = self.resampler(residual)
        if self.do_pool:
            o = self.mp(o)
            o = o + self.downsample(residual)
        else:
            o = o + residual
        o = self.a(o)

        return o


# the model for CNN-LSTM-RES with and without grid.
class SEED_DEEP(nn.Module):
    def __init__(self, do_pool=True, in_channels=1, is_grid=False, grid_size=(200, 62), out_channels=200,
                 num_classes=3, num_res_layers=5, ll1=1024, ll2=256, dropoutc=0.01, dropout1=0.5, dropout2=0.5,
                 debug=False):
        super(SEED_DEEP, self).__init__()
        self.is_grid = is_grid
        self.debug = debug
        # must use modulelist vs regular list to keep everythign in GPU and gradients flowing
        self.res_layers = nn.ModuleList()
        c = in_channels
        for r in range(num_res_layers):
            self.res_layers.append(
                SEED_DEEP_LAYER(in_channels=c, out_channels=out_channels, do_pool=do_pool, dropout=dropoutc,
                                debug=debug))
            c = out_channels
        # self.lstm1 = nn.LSTM(ll1, ll1, batch_first=True)
        # 可学习的位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, 50, ll1))
        self.transformer = TransformerBlock(dim=ll1, num_heads=4)
        self.lin1 = nn.Linear(ll1, ll2)
        self.lin2 = nn.Linear(ll2, num_classes)
        self.do1 = nn.Dropout(dropout1)
        self.do2 = nn.Dropout(dropout2)
        self.ldown = None
        if do_pool:
            self.ldown = nn.Linear(6, ll1)
        else:
            self.ldown = nn.Linear(out_channels * grid_size[0] * grid_size[1], ll1)
        self.la = nn.ReLU()

    def forward(self, x):
        # x = x.unsqueeze(1)
        # print(x.shape)
        # if it's not a grid put it in shape (batch_size, 1, 200, 62), otherwise shape is (batch_size, 200, 9, 9)
        if not self.is_grid:
            x = torch.permute(x, (0, 2, 1))
            x = x.unsqueeze(1)
        o = x  # (batch_size, 1, 200, 62)
        for i in range(len(self.res_layers)):  # 256,50,6,1
            o = self.res_layers[i](o)

        o = o.squeeze(-1)  # 去掉最后一维，转成 [B, 50, 6]
        # 4) 通道 FC → ll1 维： [B,50,6] -> [B,50,ll1]
        o = self.ldown(o)

        # o = o.view(o.shape[0], -1)  # 256 300
        # o = self.ldown(o)
        # res = o
        # o, _ = self.lstm1(o)
        # o = self.temporal_attn(o)  # <— 在这里插入时序自注意力
        # o = o + res
        # o = self.do1(o)
        # o = self.la(self.lin1(o))
        # o = self.do2(o)
        # o = self.lin2(o)
        # 6) 加位置编码
        o = o + self.pos_emb
        # 7) Transformer（包含残差+Norm+FFN）
        o = self.transformer(o)  # -> [B,50,ll1]
        # 8) 序列聚合：直接取最后一个时间步
        feat = o[:, -1, :]  # -> [B,ll1]
        # 9) 分类头
        feat = self.do1(feat)
        feat = self.la(self.lin1(feat))
        feat = self.do2(feat)
        logits = self.lin2(feat)

        return logits


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两层全连接实现“压缩—激活—重建”
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: (B, C, H, W) -> (B, C)
        y = self.avg_pool(x).view(b, c)
        # Excitation: (B, C) -> (B, C) 权重
        y = self.fc(y).view(b, c, 1, 1)
        # 重标记
        return x * y.expand_as(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        # Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, D]
        res1 = x
        attn_out, _ = self.attn(x, x, x)            # [B, T, D]
        x = self.norm1(res1 + attn_out)             # 残差 + Norm
        res2 = x
        x = self.norm2(res2 + self.ffn(x))          # FFN + 残差 + Norm
        return x
