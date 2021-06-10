import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


class LSTMWithProjection(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, proj_size, bias=False)

    def forward(self, x):
        self.lstm.flatten_parameters()
        o, (_, _) = self.lstm(x)
        return self.linear(o)

class LSTMWithoutProjection(nn.Module):
    def __init__(self, input_dim, lstm_dim, proj_dim, num_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=lstm_dim,
                            num_layers=num_lstm_layers,
                            batch_first=True)
        self.linear = nn.Linear(lstm_dim, proj_dim, bias=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.relu(self.linear(hidden[-1]))

class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim, proj_dim=256, lstm_dim=768, num_lstm_layers=3, use_lstm_with_projection=True):
        super().__init__()
        self.use_lstm_with_projection = use_lstm_with_projection
        layers = []
        # choise LSTM layer
        if use_lstm_with_projection:
            layers.append(LSTMWithProjection(input_dim, lstm_dim, proj_dim))
            for _ in range(num_lstm_layers - 1):
                layers.append(LSTMWithProjection(proj_dim, lstm_dim, proj_dim))
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = LSTMWithoutProjection(input_dim, lstm_dim, proj_dim, num_lstm_layers)

        self._init_layers()

    def _init_layers(self):
        for name, param in self.layers.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        # TODO: implement state passing for lstms
        d = self.layers(x)
        if self.use_lstm_with_projection:
            d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
        else:
            d = torch.nn.functional.normalize(d, p=2, dim=1)
        return d

    @torch.no_grad()
    def inference(self, x):
        d = self.layers.forward(x)
        if self.use_lstm_with_projection:
            d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
        else:
            d = torch.nn.functional.normalize(d, p=2, dim=1)
        return d

    def compute_embedding(self, x, num_frames=160, overlap=0.5):
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        num_overlap = int(num_frames * overlap)
        max_len = x.shape[1]
        embed = None
        cur_iter = 0
        for offset in range(0, max_len, num_frames - num_overlap):
            cur_iter += 1
            end_offset = min(x.shape[1], offset + num_frames)
            frames = x[:, offset:end_offset]
            if embed is None:
                embed = self.inference(frames)
            else:
                embed += self.inference(frames)
        return embed / cur_iter

    def batch_compute_embedding(self, x, seq_lens, num_frames=160, overlap=0.5):
        """
        Generate embeddings for a batch of utterances
        x: BxTxD
        """
        num_overlap = num_frames * overlap
        max_len = x.shape[1]
        embed = None
        num_iters = seq_lens / (num_frames - num_overlap)
        cur_iter = 0
        for offset in range(0, max_len, num_frames - num_overlap):
            cur_iter += 1
            end_offset = min(x.shape[1], offset + num_frames)
            frames = x[:, offset:end_offset]
            if embed is None:
                embed = self.inference(frames)
            else:
                embed[cur_iter <= num_iters, :] += self.inference(
                    frames[cur_iter <= num_iters, :, :]
                )
        return embed / num_iters

# adapted from https://github.com/cvqluu/GE2E-Loss
class GE2ELoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        """
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        """
        super(GE2ELoss, self).__init__()
        # pylint: disable=E1102
        self.w = nn.Parameter(torch.tensor(init_w))
        # pylint: disable=E1102
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        # print(' > Initialised Generalized End-to-End loss')

        assert self.loss_method in ["softmax", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    # pylint: disable=R0201
    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        """
        Calculates the new centroids excluding the reference utterance
        """
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1 :]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(
                    dvecs, centroids, spkr_idx, utt_idx
                )
                # vector based cosine similarity for speed
                cs_row.append(
                    torch.clamp(
                        torch.mm(
                            utterance.unsqueeze(1).transpose(0, 1),
                            new_centroids.transpose(0, 1),
                        )
                        / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)),
                        1e-6,
                    )
                )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    # pylint: disable=R0201
    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    # pylint: disable=R0201
    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat(
                    (centroids_sigmoids[:j], centroids_sigmoids[j + 1 :])
                )
                L_row.append(
                    1.0
                    - torch.sigmoid(cos_sim_matrix[j, i, j])
                    + torch.max(excl_centroids_sigmoids)
                )
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        """
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """
        centroids = torch.mean(dvecs, 1)
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = self.w * cos_sim_matrix + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.mean()

# adapted from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/angleproto.py
class AngleProtoLoss(nn.Module):
    """
    Implementation of the Angular Prototypical loss defined in https://arxiv.org/abs/2003.11982
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector
        Args:
            - init_w (float): defines the initial value of w
            - init_b (float): definies the initial value of b
    """
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(AngleProtoLoss, self).__init__()
        # pylint: disable=E1102
        self.w = nn.Parameter(torch.tensor(init_w))
        # pylint: disable=E1102
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        # print(' > Initialised Angular Prototypical loss')

    def forward(self, x):
        """
        Calculates the AngleProto loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        num_speakers = out_anchor.size()[0]

        cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1).expand(-1, -1, num_speakers), out_anchor.unsqueeze(-1).expand(-1, -1, num_speakers).transpose(0, 2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        label = torch.from_numpy(np.asarray(range(0, num_speakers))).to(cos_sim_matrix.device)
        L = self.criterion(cos_sim_matrix, label)
        return L
