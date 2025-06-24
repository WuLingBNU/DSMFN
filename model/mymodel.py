import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class gcn(nn.Module):
    def __init__(self, in_dim, out_dim, in_channel):
        super().__init__()
        self.gcn = GraphConvolution(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_channel)

    def forward(self, x, adj):
        if len(adj.shape) < 3:
            adj = adj.unsqueeze(0).repeat(x.shape[0], 1, 1)
        return self.bn(self.gcn(x, adj))


class WaveClassifier(nn.Module):
    def __init__(self, num_bands, num_rois, out_dim: int):
        super(WaveClassifier, self).__init__()

        self.num_bands = num_bands
        self.num_rois = num_rois
        self.out_dim = out_dim

        hidden_dim = num_rois  # 为了用残差，保持hidden_dim=输入时的维度
        self.after_across_dim = hidden_dim * self.num_bands * hidden_dim

        # MLP编码器 h^i = φ_MLP(A_i)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.num_rois, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        )

        # 频段邻接矩阵融合
        self.alpha = nn.Conv2d(in_channels=self.num_bands, out_channels=1, kernel_size=1)

        # 单频段图卷积层
        for i in range(num_bands):
            exec("self.gcn{}=gcn(self.num_rois, hidden_dim,self.num_rois)".format(i))

        self.gcn_cross = gcn(hidden_dim * self.num_bands, hidden_dim * self.num_bands, num_rois)

        for i in range(self.num_bands):
            exec("self.bn{}=nn.BatchNorm1d(self.num_rois)".format(i))
        self.bn_g1 = nn.BatchNorm1d(num_rois)
        self.bn_g2 = nn.BatchNorm1d(num_rois)
        self.fc = nn.Linear(self.after_across_dim, self.out_dim)

    def forward(self, x):
        batch, windows, bands, dim, dim = x.shape
        encoded_x = self.mlp(x)

        multi_x = x
        for i in range(self.num_bands):
            exec("feature{}=encoded_x[:,:,{},:,:].reshape(-1,dim,dim)".format(i, i))
            exec("feature{}=self.bn{}(feature{})".format(i, i, i))
            exec("adj{}=multi_x[:,:,{},:,:].reshape(-1,dim,dim)".format(i, i))
            exec("h{}=self.gcn{}(feature{},adj{})".format(i, i, i, i))
            exec("h{}=F.elu(h{})".format(i, i))  # [656, 116, 116]
        h_multi = []
        theta_multi = self.alpha(multi_x.reshape(-1, self.num_bands, dim, dim))
        for i in range(self.num_bands):
            exec("h_multi.append(h{})".format(i))
        h_multi = torch.concat(h_multi, dim=-1)
        theta_multi = theta_multi.squeeze(1)
        h_multi = self.bn_g1(h_multi)  # [batch*windows,num_rois,num_rois*numbands]
        h_cross = self.gcn_cross(h_multi, theta_multi)
        h_cross = self.bn_g2(h_cross)  # [656, 116, 232]
        h = h_cross.reshape(batch, windows, -1)  # shape is [batch,window,self.inputdim**2*self.numbands]

        return self.fc(h)


class TimeNoGCNBlock(nn.Module):
    def __init__(self, num_rois, window_size, out_dim: int, num_window: int = 60):
        super(TimeNoGCNBlock, self).__init__()

        self.window_size = window_size
        self.num_rois = num_rois
        self.out_dim = out_dim
        self.num_window = num_window

        self.hidden_dim = self.window_size
        self.d_model = self.num_rois * self.hidden_dim
        self.ln = nn.LayerNorm(self.d_model)

        # MLP编码器 h^i = φ_MLP(X_i)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=self.window_size, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        )
        self.mlp2 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                                  nn.Linear(self.hidden_dim, self.hidden_dim),
                                  nn.BatchNorm1d(self.num_rois))
        self.bn = nn.BatchNorm1d(self.num_rois)

        self.fc = nn.Linear(self.hidden_dim * self.num_rois, out_dim)

    def forward(self, x):
        batch, windows, dim, length = x.shape
        # 加残差
        encoded_x = (self.mlp1(x) + x).reshape(-1, dim, length)

        feature = self.bn(encoded_x)
        h = F.elu(self.mlp2(feature)) + feature

        h = h.reshape(batch, windows, -1)  # [batch,window,num_rois*length]
        x = self.fc(h)
        return x


class DualLSTMClassifier(nn.Module):
    def __init__(self, num_rois, window_size, num_bands, out_dim: int, num_window: int = 60, embedding_dim: int = 64):
        "forward接收的输入是一个tuple [raw_data,wave_matrix],形状分别为[batch,windows,rois,window_size] [batch,windows,num_bands,rois,rois]"
        super(DualLSTMClassifier, self).__init__()

        self.window_size = window_size
        self.num_rois = num_rois
        self.out_dim = out_dim
        self.num_window = num_window
        self.num_bands = num_bands
        self.embedding_dim = embedding_dim

        self.wave_block = WaveClassifier(num_bands, num_rois, embedding_dim)
        self.signal_block = TimeNoGCNBlock(num_rois, window_size, embedding_dim, num_window)
        self.fusion_block = FusionBlock(num_window, 64)
        self.d_model = embedding_dim * 2
        self.ln = nn.LayerNorm(self.d_model)

        self.lstm_layers = 2
        self.lstm_hidden = 128
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.lstm_hidden, batch_first=True,
                            num_layers=self.lstm_layers)
        self.fc = nn.Linear(self.lstm_hidden, out_dim)

    def forward(self, x):
        data, mat = x
        batch, windows, dim, length = data.shape
        mat_embedding = self.wave_block(mat)
        data_embedding = self.signal_block(data)
        h = torch.stack([data_embedding, mat_embedding], dim=1)  # 形状是[b,2,window_size,emb_dim]
        h = self.fusion_block(h)
        h = self.ln(h)

        b0 = torch.zeros(self.lstm_layers, batch, self.lstm_hidden).to(h.device)
        c0 = torch.zeros(self.lstm_layers, batch, self.lstm_hidden).to(h.device)
        out, (bn, cn) = self.lstm(h, (b0, c0))
        out = out[:, -1, :]

        x = self.fc(out)
        return x


class FusionBlock(nn.Module):
    def __init__(self, num_window, hidden_dim=64):
        super(FusionBlock, self).__init__()
        self.stream_attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(2, hidden_dim, 1), nn.ReLU(),
                                         nn.Conv2d(hidden_dim, 2, 1),
                                         nn.Softmax(1))
        self.seq_attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_window, num_window // 2, 1), nn.ReLU(),
                                      nn.Conv2d(num_window // 2,
                                                num_window, 1), nn.Softmax(1))

    def forward(self, x):
        b, channel, h, w = x.shape  # 输入形状 [b,2,window_size,emb_dim]
        s_attn = self.stream_attn(x)  # [b,2,1,1]
        seq_attn = self.seq_attn(x.transpose(1, 2))  # [b,window_size,1,1]
        attn = torch.bmm(s_attn.squeeze(-1), seq_attn.squeeze(-1).transpose(1, 2))  # [b,2,window_size]
        x = x * attn.unsqueeze(-1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, h, -1)
        return x
