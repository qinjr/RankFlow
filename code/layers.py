import numpy as np
import torch
import torch.nn.functional as F

class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, vocabulary_size, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        # maxval = np.sqrt(6. / np.sum(embed_dim))
        # minval = -maxval
        # torch.nn.init.uniform_(self.embedding.weight, minval, maxval)
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.embedding(x)


class Linear(torch.nn.Module):
    def __init__(self, vocabulary_size, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(vocabulary_size, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationLayer(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix #[B, 1]


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True, act = 'relu'):
        super().__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Dropout(p=dropout))
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            if act == 'relu':
                layers.append(torch.nn.ReLU())
            elif act == 'tanh':
                layers.append(torch.nn.Tanh())
            input_dim = hidden_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)

class HistAtt(torch.nn.Module):
    def __init__(self, q_dim):
        super().__init__()
        self.null_attention = -2 ** 32
        input_dim = 4 * q_dim # [q, k, q * k, q - k]
        layers = list()
        for hidden_dim in [200, 80]:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=0.2))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.atten_net = torch.nn.Sequential(*layers)
    
    def forward(self, x_item, user_hist, hist_len):
        _, len, dim = user_hist.shape
        x_item_tile = torch.tile(x_item.reshape([-1, 1, dim]), [1, len, 1])
        attention_inp = torch.cat((x_item_tile, user_hist, x_item_tile * user_hist, x_item_tile - user_hist), dim=2)
        score = self.atten_net(attention_inp)
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        score[~mask] = self.null_attention

        atten_score = torch.nn.Softmax(dim = 1)(score)
        user_hist_rep = torch.sum(user_hist * atten_score, dim=1)

        return user_hist_rep, atten_score.squeeze()

class HistAttProd(torch.nn.Module):
    def __init__(self, q_dim):
        super().__init__()
        self.null_attention = -2 ** 12
        self.atten_net = torch.nn.Linear(q_dim, q_dim)
    
    def forward(self, x_item, user_hist, hist_len):
        _, len, dim = user_hist.shape
        x_item_tile = torch.tile(x_item.reshape([-1, 1, dim]), [1, len, 1])
        q = self.atten_net(x_item_tile)
        
        score = torch.sum(q * user_hist, dim=2, keepdim=True)
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        score[~mask] = self.null_attention

        atten_score = torch.nn.Softmax(dim = 1)(score)
        user_hist_rep = torch.sum(user_hist * atten_score, dim=1)

        return user_hist_rep, atten_score.squeeze()

class RNN(torch.nn.Module):
    def __init__(self, lr, input_size, hidden_size, mlp_hidden_dims, output_shape, wd, drop,
                batch_size, device = 'cuda:0'):
        super(RNN, self).__init__()
        self.device = device

        # GRU
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first = True, bidirectional = True).to(device)

        # MLP
        layers = list()
        input_dim = 2 * hidden_size
        for hidden_dim in mlp_hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=drop))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim

        layers.append(torch.nn.Linear(input_dim, output_shape))
        self.mlp = torch.nn.Sequential(*layers).to(device)

        self.loss_fun = torch.nn.NLLLoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, x_batch):
        _, h_T = self.gru(x_batch)
        h_T = h_T.squeeze()
        h_T = torch.cat((h_T[0], h_T[1]), dim=1)
        out = torch.nn.Softmax(dim=1)(self.mlp(h_T)) #[B, OutSize]
        return out