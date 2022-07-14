import torch
import torch.nn as nn

class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float): Small value to avoid division by zero
    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    Examples::
        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class BPR_LogLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float): Small value to avoid division by zero
    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    Examples::
        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPR_LogLoss, self).__init__()
        self.gamma = gamma
        self.bce_loss_func = torch.nn.BCELoss()

    def forward(self, pos_score, neg_score):
        labels_pos = torch.ones_like(pos_score).to(pos_score.device)
        labels_neg = torch.ones_like(neg_score).to(neg_score.device)
        preds = torch.cat((pos_score, neg_score))
        labels = torch.cat((labels_pos, labels_neg))
        
        loss = self.bce_loss_func(preds, labels)
        return loss

class RegLoss(nn.Module):
    """ RegLoss, L2 regularization on model parameters
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss

class TopKLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(TopKLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, pred_s, pred_t, k, list_len):
        pred_s = pred_s.reshape([-1, list_len])
        pred_t = pred_t.reshape([-1, list_len])
        
        idx = torch.argsort(pred_t, dim=1, descending=True)
        sorted_pred_s = torch.gather(pred_s, dim=1, index=idx)
        topk_scores = torch.mean(sorted_pred_s[:,:k], dim=1)
        no_topk_scores = torch.mean(sorted_pred_s[:,k:], dim=1)
        loss = -torch.log(self.gamma + torch.sigmoid(topk_scores - no_topk_scores)).mean()
        return loss

class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self.bce_loss_func = torch.nn.BCELoss()
        self.softmax_func = torch.nn.Softmax(dim=1)
    
    def forward(self, pred, label, batch_neg_size):
        pred = torch.transpose(pred.reshape((batch_neg_size + 1, -1)), 0, 1)
        label = torch.transpose(label.reshape((batch_neg_size + 1, -1)), 0, 1)
        pred_softmax = self.softmax_func(pred)
        loss = self.bce_loss_func(pred_softmax, label)
        return loss

class ICCLoss(nn.Module):
    def __init__(self):
        super(ICCLoss, self).__init__()
        self.bce_loss_func = torch.nn.BCELoss()
    
    def forward(self, model_preds, label):
        indicators = None
        for i in range(model_preds.shape[1]):
            k = torch.tanh(torch.tensor([i])).to(model_preds.device)
            # k = torch.tensor([i]).to(model_preds.device)
            indicator = torch.sigmoid(model_preds[:,i] - k).unsqueeze(1)
            indicators = indicator if indicators == None else torch.cat((indicators, indicator), dim=1)
        weights = torch.cumprod(torch.cat((torch.ones((indicators.shape[0],1)).to(indicators.device), indicators), dim=1), dim=1)[:,:-1]
        weights = (1 - weights) * weights
        score = torch.sum(model_preds * weights, dim=1)
        loss = self.bce_loss_func(score, label)
        return loss