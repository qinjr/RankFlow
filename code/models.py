import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layers import FeaturesEmbedding
from layers import Linear
from layers import FactorizationLayer
from layers import MultiLayerPerceptron
from layers import InnerProductNetwork
from layers import HistAtt, HistAttProd

class Rec(torch.nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.vocabulary_size = data_config['vocabulary_size']
        self.user_num_fields = data_config['user_num_fields']
        self.item_num_fields = data_config['item_num_fields']
        self.num_fields = data_config['num_fields']
        self.embed_dim = model_config['embed_dim'] if 'embed_dim' in model_config else None
        self.hidden_dims = model_config['hidden_dims'] if 'hidden_dims' in model_config else None
        self.dropout = model_config['dropout'] if 'dropout' in model_config else None
        self.use_hist = model_config['use_hist'] if 'use_hist' in model_config else None
        self.batch_random_neg = model_config['batch_random_neg'] if 'batch_random_neg' in model_config else None


class LR(Rec):
    """
    A pytorch implementation of Logistic Regression.
    """
    def __init__(self, model_config, data_config):
        super(LR, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
    
    def get_name(self):
        return 'LR'

    def forward(self, x_user, x_item, user_hist = None, hist_len = None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat((x_user, x_item), dim=1)
        return torch.sigmoid(self.linear(x).squeeze(1))


class DSSM(Rec):
    """
    A pytorch implementation of DSSM as recall model, plain dual tower DNN.
    """

    def __init__(self, model_config, data_config):
        super(DSSM, self).__init__(model_config, data_config)
        self.user_vec_dim = self.user_num_fields * self.embed_dim
        self.item_vec_dim = self.item_num_fields * self.embed_dim

        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.user_tower = MultiLayerPerceptron(self.user_vec_dim, self.hidden_dims, self.dropout, False, 'tanh')
        self.item_tower = MultiLayerPerceptron(self.item_vec_dim, self.hidden_dims, self.dropout, False, 'tanh')

    def get_name(self):
        return 'DSSM'

    def forward(self, x_user, x_item, user_hist = None, hist_len = None):
        self.user_emb = self.embedding(x_user).view(-1, self.user_vec_dim)
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        self.u = self.user_tower(self.user_emb)
        self.i = self.item_tower(self.item_emb)
        score = torch.sigmoid(torch.sum(self.u * self.i, dim=1, keepdim=False))
        # score = torch.sigmoid(torch.cosine_similarity(self.u, self.i, dim=1))
        return score

    def get_user_repre(self, x_user, user_hist = None, hist_len = None):
        self.user_emb = self.embedding(x_user).view(-1, self.user_vec_dim)
        return self.user_tower(self.user_emb)

    def get_item_repre(self, x_item):
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        return self.item_tower(self.item_emb)


class COLD(Rec):
    """
    A pytorch implementation of COLD as pre-ranking model
    """
    def __init__(self, model_config, data_config):
        super(COLD, self).__init__(model_config, data_config)
        self.user_vec_dim = self.user_num_fields * self.embed_dim
        self.item_vec_dim = self.item_num_fields * self.embed_dim
        self.cross_vec_dim = int(self.num_fields*(self.num_fields-1)/2)

        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.inner_product = InnerProductNetwork()
        self.all_tower = MultiLayerPerceptron(self.user_vec_dim+self.item_vec_dim+self.cross_vec_dim, self.hidden_dims,self.dropout, True)

    def get_name(self):
        return 'COLD'

    def forward(self, x_user, x_item, user_hist = None, hist_len = None):
        x = torch.cat((x_user, x_item), dim=1)
        self.user_emb = self.embedding(x_user).view(-1, self.user_vec_dim)
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        self.all_emb = self.embedding(x)
        self.cross = self.inner_product(self.all_emb)

        self.all_emb = torch.cat((self.user_emb,self.item_emb,self.cross),1)

        score = torch.sigmoid(self.all_tower(self.all_emb).squeeze(1))

        return score

class YouTubeDNN(Rec):
    """
    A pytorch implementation of DSSM as recall model, plain dual tower DNN.
    """

    def __init__(self, model_config, data_config):
        super(YouTubeDNN, self).__init__(model_config, data_config)
        self.user_vec_dim = (self.user_num_fields + self.item_num_fields) * self.embed_dim
        self.item_vec_dim = self.item_num_fields * self.embed_dim

        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.user_tower = MultiLayerPerceptron(self.user_vec_dim, self.hidden_dims, self.dropout, False, 'tanh')
        self.item_tower = MultiLayerPerceptron(self.item_vec_dim, self.hidden_dims, self.dropout, False, 'tanh')

    def get_name(self):
        return 'YouTubeDNN'

    def forward(self, x_user, x_item, user_hist, hist_len):
        self.user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        
        self.user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        # get mask
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        self.user_hist[~mask] = 0.0

        self.user_hist_rep = torch.mean(self.user_hist, dim=1)
        self.u = self.user_tower(torch.cat((self.user_emb, self.user_hist_rep), dim=1))
        self.i = self.item_tower(self.item_emb)
        score = torch.sigmoid(torch.sum(self.u * self.i, dim=1, keepdim=False))
        # score = torch.sigmoid(torch.cosine_similarity(self.u, self.i, dim=1))
        return score

    def get_user_repre(self, x_user, user_hist, hist_len):
        self.user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        self.user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        # get mask
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        self.user_hist[~mask] = 0.0
        
        self.user_hist_rep = torch.mean(self.user_hist, dim=1)
        return self.user_tower(torch.cat((self.user_emb, self.user_hist_rep), dim=1))

    def get_item_repre(self, x_item):
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        return self.item_tower(self.item_emb)


class WDL(Rec):
    """
    A pytorch implementation of wide and deep learning.
    """

    def __init__(self, model_config, data_config):
        super(WDL, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.input_dim = self.num_fields * self.embed_dim
        self.mlp = MultiLayerPerceptron(self.input_dim, self.hidden_dims, self.dropout)
    
    def get_name(self):
        return 'WDL'
    
    def forward(self, x_user, x_item, user_hist = None, hist_len = None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.input_dim))
        return torch.sigmoid(x.squeeze(1))

class FM(Rec):
    def __init__(self, model_config, data_config):
        super(FM, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
        self.fm = FactorizationLayer(reduce_sum=True)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
    
    def get_name(self):
        return 'FM'
    
    def forward(self, x_user, x_item, user_hist = None, hist_len = None):
        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x)
        return torch.sigmoid(x.squeeze(1))

class DeepFM(Rec):
    """
    A pytorch implementation of DeepFM.
    """
    def __init__(self, model_config, data_config):
        super(DeepFM, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
        self.fm = FactorizationLayer(reduce_sum=True)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.input_dim = self.num_fields * self.embed_dim
        self.mlp = MultiLayerPerceptron(self.input_dim, self.hidden_dims, self.dropout)

    def get_name(self):
        return 'DeepFM'

    def forward(self, x_user, x_item, user_hist = None, hist_len = None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.input_dim))
        return torch.sigmoid(x.squeeze(1))

class DIN(Rec):
    """
    A pytorch implementation of DSSM as recall model, plain dual tower DNN.
    """

    def __init__(self, model_config, data_config):
        super(DIN, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = (self.user_num_fields + 2 * self.item_num_fields) * self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att = HistAttProd(self.item_num_fields * self.embed_dim)
        
    def get_name(self):
        return 'DIN'

    def forward(self, x_user, x_item, user_hist, hist_len):
        user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)

        user_rep, atten_score = self.att(item_emb, user_hist, hist_len)
        inp = torch.cat((user_emb, item_emb, user_rep), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out)
    

class DIEN(Rec):
    def __init__(self, model_config, data_config):
        super().__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = (self.user_num_fields + 2 * self.item_num_fields) * self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att = HistAttProd(self.item_num_fields * self.embed_dim)
        self.gru = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.item_num_fields * self.embed_dim, batch_first=True)
    
    def get_name(self):
        return 'DIEN'
    
    def forward(self, x_user, x_item, user_hist, hist_len):
        user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        hist_len = torch.where(hist_len > 0, hist_len, torch.ones_like(hist_len))
        
        hist_len_c = hist_len.cpu()
        user_hist_p = pack_padded_sequence(user_hist, hist_len_c, batch_first=True, enforce_sorted=False)
        hidden_seq, _ = self.gru(user_hist_p)
        hidden_seq, _ = pad_packed_sequence(hidden_seq, batch_first=True)

        user_rep, atten_score = self.att(item_emb, hidden_seq, hist_len)
        inp = torch.cat((user_emb, item_emb, user_rep), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out)
