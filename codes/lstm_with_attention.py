import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import typing as tp
import opensmile

from tqdm.notebook import tqdm
from .base import BaseClassificationModel
from .data_prep import BaseCustomDataset


class TimeAttention(nn.Module):
    """
    Time-Attention module.

    Input: `tokens` with the shape (`batch_size`, `time_dim`, `features_dim`)

    `w` is a parameter tensor of shape (`features_dim`, `features_dim`) 

    Output shape is (`batch_size`, `features_dim`)
    This module works somehow like the following:

    >>> query = tokens[-1:, :]  # 1 query
    >>> key = tokens @ w  # `time_dim` keys and values
    >>> value = tokens
    >>> attention = attention(query, key, value)
    >>> return attention.unsqueeze(1)
    """
    w: nn.Parameter

    def __init__(self, feature_dim):
        super().__init__()
        self.w = nn.Parameter(torch.empty(feature_dim, feature_dim))

        nn.init.xavier_normal_(self.w.data)

    def forward(self, x:torch.Tensor):
        # x.shape = (batch_size, time_dim, features_dim)
        att_logits = (x[:,-1:, :] @ self.w[None,:,:]) @ x.transpose(-2, -1)
        # att_logits.shape == (batch_size, 1, time_dim)
        att_weigts = torch.softmax(att_logits, -1)
        outs = att_weigts @ x
        # outs.shape = (batch_size, 1, features_dim)
        return outs.squeeze(1)


class FeatureAttention(nn.Module):
    """
    Feature-Attention module.

    Input: `tokens` with the shape (`batch_size`, `time_dim`, `features_dim`)

    `w`, `v` are parameter tensor of shape (`features_dim`, `features_dim`) 

    Output shape is (`batch_size`, `features_dim`)
    This module works somehow like the following:

    >>> query = v.T.unsqueeze(-1)  # `features_dim` trainable queries
    >>> key = torch.tanh(tokens @ w)  # `time_dim` keys and vlues
    >>> value = features
    >>> attention = attention(query, key, value)
    >>> return torch.diagonal(attention, dim1=-1, dim2=-2)
    """
    w: nn.Parameter
    def __init__(self, feature_dim) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.empty(feature_dim, feature_dim))
        self.v = nn.Parameter(torch.empty(feature_dim, feature_dim))

        nn.init.xavier_normal_(self.w.data)
        nn.init.xavier_normal_(self.v.data)

    def forward(self, x:torch.Tensor):
        # x.shape = (batch_size, time_dim, features_dim)
        att_logits = torch.tanh(x @ self.w) @ self.v
        # att_logits.shape == (batch_size, time_dim, features_dim)
        att_weigts = torch.softmax(att_logits, -2)
        outs = (att_weigts * x).sum(1)
        # outs.shape = (batch_size, features_dim)
        return outs


class AttentionLSTM_Dataset(BaseCustomDataset):
    def __init__(self, *args, num_workers=2, **kwargs):
        super().__init__(*args, **kwargs)

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            num_workers=num_workers
        )
        self.smile_de = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
            num_workers=num_workers
        )

        self.features = {}
        mode = "val" if len(self) < 3000 else "train"
        try:
            with open(f"ComParE_2016_{mode}.npz", "rb") as file:
                features_lst = np.load(file)["arr_0"]
        except FileNotFoundError:
            features_lst = np.stack([self.extract_features(path) for path, _, _ in tqdm(self.data_list)], 0)
            with open(f"ComParE_2016_{mode}.npz", "wb") as file:
                np.savez(file, features_lst)
        self.features = {path:feature for (path, _, _), feature in zip(self.data_list, features_lst)}


    def extract_features(self, path):
        outs = self.features.get(path, None)
        if outs is None:
            x = self.load_wav(path)
            features = torch.tensor(self.smile.process_signal(x, self.sampling_rate).to_numpy())
            features_de = torch.tensor(self.smile_de.process_signal(x, self.sampling_rate).to_numpy()[:-2])
            return torch.cat([features, features_de], 1)
        else:
            return outs



class AttentionLSTM_Classifier(BaseClassificationModel):
    def __init__(self, input_dims=93, lstms:tp.Sequence[nn.Module]=None, context_dims=12):
        super().__init__()
        lstms_out_dims = 256
        if lstms is None:
            self.lstm1 = nn.LSTM(batch_first=True,
                                 input_size=input_dims,
                                 hidden_size=512)
            self.lstm2 = nn.LSTM(batch_first=True,
                                 input_size=512,
                                 hidden_size=lstms_out_dims)
        else:
            self.lstm1, self.lstm2 = lstms

        self.time_attention = TimeAttention(lstms_out_dims)
        self.feature_attention = FeatureAttention(lstms_out_dims)

        classifier_in_dims = 2*lstms_out_dims + context_dims
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x:torch.Tensor, context:tp.Optional[torch.Tensor]=None):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = torch.cat([
            self.time_attention(x),
            self.feature_attention(x),
            context,
        ], 1)

        x = self.classifier(x)
        
        return x
