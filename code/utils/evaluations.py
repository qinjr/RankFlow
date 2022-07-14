import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

'''
Top-K metrics including:
HR, NDCG, MRR, MAP, Recall, Precision, F1-score
labels and preds are in full-batch mode
'''
class TopKMetric(object):
    def __init__(self, topks: list, list_len: int, labels: np.ndarray, preds: np.ndarray, bids: np.ndarray,
                total_relevant_num = None, mode = None) -> None:
        super().__init__()
        self.topks = topks
        self.metrics = {}
        # get sorted labels w.r.t preds
        self.list_len = list_len
        self.labels = labels.reshape([-1, self.list_len])
        self.preds = preds.reshape([-1, self.list_len])
        self.sorted_index = np.argsort(-self.preds, axis=1)
        self.sorted_labels = np.take_along_axis(self.labels, self.sorted_index, axis=1)
        self.bids = bids
        if bids is not None:
            self.sorted_bids = np.take_along_axis(self.bids, self.sorted_index, axis=1)
        
        self.pos = np.array(list(range(1, self.list_len + 1)) * self.sorted_labels.shape[0]).reshape([-1, self.list_len])
        
        self.inf_vec = np.ones([self.labels.shape[0],]) * np.inf
        self.total_relevant_num = total_relevant_num
        self.mode = mode


    def cal_HR(self):
        for k in self.topks:
            hr_k = np.average(np.sum(self.sorted_labels[:,:k], axis=1) / k, axis=0)
            self.metrics['HR@{}'.format(k)] = hr_k
        
    def cal_nDCG(self):
        idea_labels = -np.sort(-self.labels, axis=1)
        for k in self.topks:
            dcg_k = np.sum(self.sorted_labels[:,:k] / np.log2(self.pos[:,:k] + 1), axis=1)
            idcg_k = np.sum(idea_labels[:,:k] / np.log2(self.pos[:,:k] + 1), axis=1)
            idcg_k = np.where(idcg_k == 0, self.inf_vec, idcg_k)
            ndcg_k = np.average(dcg_k / idcg_k, axis=0)
            self.metrics['nDCG@{}'.format(k)] = ndcg_k

    def cal_MRR(self):
        summed = np.sum(self.sorted_labels, axis=1)
        position_of_relevant = np.where(summed == 0, self.inf_vec, np.argmax(self.sorted_labels, axis=1) + 1)
        mrr = np.average(1 / position_of_relevant, axis=0)
        self.metrics['MRR'] = mrr
    
    def cal_Precision(self):
        for k in self.topks:
            precision_k = np.average(np.sum(self.sorted_labels[:,:k], axis=1) / k, axis=0)
            self.metrics['Precision@{}'.format(k)] = precision_k
    
    def cal_Recall(self):
        total_relevant = self.total_relevant_num
        if total_relevant is None:
            total_relevant = np.sum(self.labels, axis=1)
        total_relevant = np.where(total_relevant == 0, self.inf_vec, total_relevant)
        for k in self.topks:
            recall_k = np.average(np.sum(self.sorted_labels[:,:k], axis=1) / total_relevant, axis=0)
            self.metrics['Recall@{}'.format(k)] = recall_k

    def cal_MAP(self):
        precision_at_each_pos = np.cumsum(self.sorted_labels, axis=1) / self.pos
        label_summed = np.sum(self.labels, axis=1, keepdims=True)
        label_summed = np.where(label_summed == 0, np.expand_dims(self.inf_vec, 1), label_summed)
        ap = np.cumsum(precision_at_each_pos * self.sorted_labels, axis=1) / label_summed
        for k in self.topks:
            map_k = np.average(ap[:,k-1])
            self.metrics['MAP@{}'.format(k)] = map_k
    
    def cal_F1(self):
        for k in self.topks:
            p_k = self.metrics['Precision@{}'.format(k)]
            r_k = self.metrics['Recall@{}'.format(k)]
            self.metrics['F1@{}'.format(k)] = 2 * p_k * r_k / (p_k + r_k)
    
    def cal_CTR(self):
        total_clicks = np.sum(self.labels)
        total_pv = np.size(self.labels)
        self.metrics['CTR'] = total_clicks / total_pv
    
    def cal_eCPM(self):
        total_pv = np.size(self.labels)
        pay = self.sorted_bids[:,1:]
        pay = np.concatenate((pay, np.zeros((pay.shape[0], 1))), axis=1)
        total_income = np.sum(self.sorted_labels * pay)
        self.metrics['eCPM'] = total_income * 1000 / total_pv

    def cal_ARPU(self):
        pay = self.sorted_bids[:,1:]
        pay = np.concatenate((pay, np.zeros((pay.shape[0], 1))), axis=1)
        total_income = np.sum(self.sorted_labels * pay)
        self.metrics['ARPU'] = total_income / pay.shape[0]


    def get_metrics(self):
        if self.mode == None:
            self.cal_HR()
            self.cal_nDCG()
            self.cal_MRR()
            self.cal_Recall()
            self.cal_MAP()
        elif self.mode == 'recall':
            self.cal_Precision()
            self.cal_Recall()
            self.cal_F1()
        elif self.mode in ['ranking', 'pre-ranking', 're-ranking']:
            self.cal_HR()
            self.cal_nDCG()
            self.cal_MRR()
            self.cal_MAP()
            if self.bids is not None:
                self.cal_CTR()
                self.cal_eCPM()
                self.cal_ARPU()
        return self.metrics


'''
Point metrics including:
AUC, LL
labels and preds are in full-batch mode
'''
class PointMetric(object):
    def __init__(self, labels: np.ndarray, preds: np.ndarray) -> None:
        super().__init__()
        self.metrics = {}
        self.labels = labels
        self.preds = preds


    def cal_AUC(self):
        self.metrics['AUC'] = roc_auc_score(self.labels, self.preds)
        
    def cal_LL(self):
        self.metrics['LL'] = log_loss(self.labels, self.preds.astype("float64"))
    
    def get_metrics(self):
        self.cal_AUC()
        self.cal_LL()
        return self.metrics
