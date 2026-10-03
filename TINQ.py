"""Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
#
# Author: YIXIAO Li
# Email:
# Date:
# CUDA_VISIBLE_DEVICES=1 python timepool2.py --database=KoNViD-1k --exp_id=0

from argparse import ArgumentParser
import os
import h5py
from torch.optim import Adam,SGD, lr_scheduler
from torch.utils.data import Dataset
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime
from scipy.stats import spearmanr, kendalltau, pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix


class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/',features_dir2='CNN_features_KoNViD-1k/', index=None, max_len=240, feat_dim=4096,feat_dim2=1536, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.features2 = np.zeros((len(index), max_len, feat_dim2))
        self.features3 = np.zeros((len(index), max_len, feat_dim2+feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        self.coms = np.zeros((len(index), max_len))
        #self.threshold = np.zeros((len(index), 1))
        #print(len(index))
        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + '_resnet-50_res5c.npy')
            features2 = np.load(features_dir2 + str(index[i]) + '_resnet-50_res5c.npy')
            features3 = np.hstack((features.squeeze(),features2.squeeze()))
            self.length[i] = features.shape[0]
            self.features3[i, :features.shape[0], :] = features3
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  #
            self.coms[i, :np.int(self.length[i])] = torch.ones(np.int(self.length[i])) # For convenience, the complexity is set to 1. Pls compute the complexity by running complexitynew.py and load it here.
        self.threshold = compute_threshold(self.coms,self.length)
        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        # 获取特征和标签
        features = self.features3[idx]
        label = self.label[idx]
        coms = self.coms[idx]
        thres = self.threshold[idx]
        actual_len = self.length[idx]

        return features, coms, thres,label,actual_len

def compute_threshold(complexity,actlen):
    thresholds = []
    level = []
    for idx in range(len(complexity)):
        complex = complexity[idx][:int(actlen[idx])]
        mean = np.mean(complex)
        std = np.std(complex)
        level.append(mean+std)
    thresholds = (5-4*(level-np.min(level))/(np.max(level)-np.min(level)+1e-6)+1e-3)


    return thresholds

# 计算邻接矩阵
def compute_adjacency_matrix(data, threshold=0.5):
    # data: shape [H, W]
    distances = euclidean_distances(data.cpu().detach().numpy())
    adjacency_matrix = np.exp(-distances**2 / (2. * threshold**2))
    return torch.tensor(adjacency_matrix, dtype=torch.float).cuda()

# 计算稀疏邻接矩阵
def compute_sparse_adjacency_matrix(data, k=5, threshold=0.5):
    # data: shape [H, W]
    distances = euclidean_distances(data.cpu().detach().numpy())
    adjacency_matrix = np.exp(-distances ** 2 / (2. * threshold ** 2))

    # 保留每行中距离最近的k个节点
    for i in range(adjacency_matrix.shape[0]):
        sorted_indices = np.argsort(adjacency_matrix[i])  # 从小到大排序
        adjacency_matrix[i, sorted_indices[:-k]] = 0  # 仅保留前k个最相似的元素

    # 将稀疏矩阵转换为CSR格式，以便进一步加速计算
    sparse_adjacency_matrix = csr_matrix(adjacency_matrix)

    # 将CSR矩阵转换为稀疏张量
    sparse_adjacency_tensor = torch.tensor(sparse_adjacency_matrix.toarray(), dtype=torch.float).cuda()

    return sparse_adjacency_tensor


# 定义GAT层
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        #print(h.shape)
        #adj = compute_adjacency_matrix(h)
        adj = compute_sparse_adjacency_matrix(h)
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


# 自适应时间片段聚合，支持多尺度聚合
class AdaptiveTemporalAggregation(nn.Module):
    def __init__(self, base_segment_size=20):
        super(AdaptiveTemporalAggregation, self).__init__()
        self.base_segment_size = base_segment_size

    def forward(self, flow_complexity,threshold):
        complexity = flow_complexity

        #fixed threshold
        threshold =1.1

        segments = []
        current_segment = []
        current_complexity = 0

        # 根据复杂度自适应划分时间片段
        for i, comp in enumerate(complexity):
            current_complexity += comp
            current_segment.append(i)

            if  current_complexity >= threshold:
                segments.append(current_segment)
                current_segment = []
                current_complexity = 0

        #if current_segment:
        #        segments.append(current_segment)

        return segments


# 自注意力机制
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(x.size(-1))
        attention_weights = self.softmax(scores)
        attended = torch.matmul(attention_weights, V)
        return attended, attention_weights


# 时间模块，结合GAT、GRU与自注意力
class TemporalModel(nn.Module):
    def __init__(self, hidden_size, num_classes, reduced_size):
        super(TemporalModel, self).__init__()
        self.fc0 = nn.Linear(5632, reduced_size)  # Ensure input feature size is correct
        self.gat_layer1 = GATLayer(2*reduced_size, 2 * hidden_size,dropout=0.6, alpha=0.2)
        self.gat_layer2 = GATLayer(2 * hidden_size, hidden_size,dropout=0.6, alpha=0.2)
        self.gru1 = nn.GRU(2 * hidden_size, 2 * hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(2*hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.temporal_agg = AdaptiveTemporalAggregation()
        self.attention1 = TemporalAttention(2*hidden_size)

    def forward(self, video_framess, flow_complexs,actlens,threshold):
        device = video_framess.device
        batch_size = video_framess.size(0)
        aggregated_features_list = []
        for b in range(batch_size):
            #print(video_framess[b].shape)
            video_frames = video_framess[b][:int(actlens[b])]
            flow_complex = flow_complexs[b][:int(actlens[b])]
            #print(flow_complex.shape)
            segments = self.temporal_agg(flow_complex,threshold[b])
            #print(video_frames.shape)
            video_frames = F.relu(self.fc0(video_frames))
            video_frames = F.dropout(video_frames, 0.5)

            aggregated_features = []

            for segment in segments:
                segment_features = [video_frames[i] for i in segment]
                segment_features = torch.stack(segment_features).to(device)
                # Compute mean or variance for aggregation
                mean_feature = segment_features.mean(dim=0)
                std_feature = segment_features.std(dim=0)
                aggregated_features.append(torch.cat([mean_feature,std_feature]))
                #aggregated_features.append(std_feature)
            aggregated_features = torch.stack(aggregated_features).to(device)
            #print(aggregated_features.shape)

            # 计算邻接矩阵
            #adj_matrix = compute_adjacency_matrix(aggregated_features).to(device)
            # 应用图注意力层
            #print(aggregated_features.shape)
            gat_output = self.gat_layer1(aggregated_features)
            gat_output = F.elu(gat_output)
            gat_output = F.dropout(gat_output, 0.3)
            #print(gat_output.shape)
            gru_output1, _ = self.gru1(gat_output.unsqueeze(0))

            # 第二阶段: 基于注意力分数的时间片段重新划分
            _, attention_weights = self.attention1(gat_output)
            attention_weights = attention_weights.squeeze(0)
            #print(attention_weights.shape)

            # 重新划分片段
            num_segments = len(attention_weights)
            attention_weights = attention_weights.softmax(dim=1)  # Normalize weights
            num_new_segments = max(1, 9*num_segments // 10)  # Set a reasonable number of new segments
            _, new_segment_indices = torch.topk(attention_weights, num_new_segments, dim=1)

            new_segments = []

            new_segments.append([i for i in new_segment_indices[0]])

            # 聚合新片段特征
            final_aggregated_features = []
            for segment in new_segments:
                segment_features = [gat_output[i] for i in segment]
            final_aggregated_features = torch.stack(segment_features).to(device)
            #print(final_aggregated_features.shape)

            # 再次应用图注意力层
            #final_adj_matrix = compute_adjacency_matrix(final_aggregated_features).to(device)
            final_gat_output = self.gat_layer2(final_aggregated_features)
            #print(final_gat_output.shape)
            final_gat_output = F.elu(final_gat_output)
            final_gat_output = F.dropout(final_gat_output, 0.3)

            # 时间关系建模（GRU）并引入自注意力机制
            gru_output2, _ = self.gru2(final_gat_output.unsqueeze(0))
            #print(gru_output.shape)

            # 回归到质量分数
            output1 = self.fc1(gru_output1.squeeze(0)).mean()
            output2 = self.fc2(gru_output2.squeeze(0)).mean()
            output = 0.5*output1+0.5*output2
            aggregated_features_list.append(output)

        return torch.stack(aggregated_features_list)


# 示例输入
#T = 10  # 假设视频有10帧
#F = 128  # 每帧特征维度
#W, H, C = 64, 64, 2  # 假设光流帧尺寸为64x64，2个通道

#video_frames = torch.randn(T, F)  # 模拟视频特征
#flow_video = np.random.randn(T, W, H, C)  # 模拟光流视频


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions, targets):
        # 计算 L1 损失
        l1_loss = self.l1_loss(predictions, targets)
        mse_loss = self.mse(predictions, targets)

        # 计算 SROCC 和 KROCC
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        srocc, _ = spearmanr(predictions_np, targets_np)
        plcc,_ = pearsonr(predictions_np, targets_np)

        # 由于我们希望最大化 SROCC 和 KROCC，所以取负值作为惩罚项
        srocc_loss = 1-srocc
        plcc_loss = 1-plcc

        # 组合损失
        #total_loss = l1_loss +sf.alpha*srocc_loss+self.beta*plcc_loss
        total_loss = mse_loss+srocc_loss
        return total_loss



if __name__ == "__main__":
    parser = ArgumentParser(description='"GATRU: Quality Assessment of In-the-Wild Videos')
    parser.add_argument("--seed", type=int, default=19920518)
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--database', default='VSR-QAD', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--model', default='GATRU', type=str,
                        help='model name (default: GATRU)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='val ratio (default: 0.2)')

    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay (default: 0.0)')

    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    args.decay_interval = int(args.epochs/10)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        features_dir = 'C:/Users/scmyl7/Desktop/myBVQA/CNN_features_KoNViD-1k/'  # features dir
        features_dir2 = 'C:/Users/scmyl7/Desktop/myBVQA/CNN_adaourswinonly_KoNViD-1k/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'LIVE-Qualcomm':
        features_dir = 'CNN_adaourCNNonly_LIVE-Qualcomm/'
        features_dir2 = 'CNN_adaourswinonly_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'
    if args.database == 'LIVE-VQC':
        features_dir = 'CNN_adaourCNNonly_LIVE-VQC/'
        features_dir2 = 'CNN_adaourswinonly_LIVE-VQC/'
        datainfo = 'data/LIVEVQCinfo.mat'
    if args.database == 'VSR-QAD':
        features_dir = 'C:/D/VQA/myBVQA/CNN_adaourCnnonlyhigh_VSR/'
        features_dir2 = 'C:/D/VQA/myBVQA/CNN_stemourswin_VSR/'
        datainfo = 'data/VSRinfo.mat'
    print('EXP ID: {}'.format(args.exp_id))
    print(args.database)
    print(args.model)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')  # index, ref_ids
    scale = Info['scores'][0, :].max()
    index = Info['index']
    index = index[:, args.exp_id % index.shape[1]]  # np.random.permutation(N)
    ref_ids = Info['ref_ids'][0, :]  #
    max_len = int(Info['max_len'][0])
    trainindex = index[0:int(np.ceil((1- args.test_ratio - args.val_ratio) * len(index)))]
    valindex = index[int(np.ceil((1 - args.val_ratio) * len(index))):len(index)]
    train_index, val_index, test_index = [], [], []
    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            val_index.append(i) if (ref_ids[i] in valindex) else \
                test_index.append(i)

    scale = Info['scores'][0, :].max()  # label normalization factor
    train_dataset = VQADataset(features_dir, features_dir2,train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = VQADataset(features_dir, features_dir2,val_index, max_len, scale=scale)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
    if args.test_ratio > 0:
        test_dataset = VQADataset(features_dir, features_dir2,test_index, max_len, scale=scale)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    # 定义GAT层参数
    reduced_size = 256
    hidden_size = 16
    num_classes = 1

    # 初始化模型
    model = TemporalModel(hidden_size, num_classes,reduced_size).to(device)

    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models/{}-{}-EXP{}-{}'.format(args.model, args.database, args.exp_id,'adaourCNNhswinl_dot9K')
    if not os.path.exists('results'):
        os.makedirs('results')
    save_result_file = 'results/{}-{}-EXP{}-{}'.format(args.model, args.database, args.exp_id,'adaourCNNhswinl_dot9K')

    if not args.disable_visualization:  # Tensorboard Visualization
        writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}-{}'
                               .format(args.log_dir, args.exp_id, args.database, args.model,
                                       args.lr, args.batch_size, args.epochs,
                                       datetime.datetime.now().strftime("%I%M%pon%B%d%Y")))
    criterion = CombinedLoss(alpha=1, beta=1)  # default L1 loss
    criterion1 = nn.L1Loss()  # L1 loss
    criterion2 = nn.MSELoss()  # L1 loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    best_val_criterion = -1  # SROCC min
    for epoch in range(args.epochs):
        #checkpoint = torch.load('C:/D/VQA/Swin-Transformer-main/models/GATRU-VSR-orgCNN')
        #new_pth = model.state_dict()  # 需要加载参数的模型
        #pretrained_dict = {}  # 用于保存公共具有的参数
        #for k, v in checkpoint.items():
        #    for kk in new_pth.keys():
        #        if kk in k:
        #            pretrained_dict[kk] = v
        #            break
        #new_pth.update(pretrained_dict)
        # Train
        model.train()
        L = 0
        for i, (features, complexity,thres,label,actlen) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            complexity = complexity.to(device).float()
            actlen = actlen.to(device).float()
            thres = thres.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features, complexity,actlen,thres)
            loss = criterion1(outputs, label.squeeze(1))
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #print(i)
            optimizer.step()
            L = L + loss.item()
        train_loss = L / (i + 1)
        print(train_loss)

        torch.cuda.empty_cache()

        model.eval()
        # Val
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0
        with torch.no_grad():
            for i, (features,fixedframe,fixthres,label,actlen) in enumerate(val_loader):
                y_val[i] = scale *label.item()  #
                features = features.to(device).float()
                #print(features.shape)
                label = label.to(device).float()
                actlen = actlen.to(device).float()
                complexity = fixedframe.to(device).float()
                fixthres = fixthres.to(device).float()
                #print(complexity.shape)
                outputs = model(features, complexity,actlen,fixthres)
                y_pred[i] = scale *outputs.item()
                loss = criterion1(outputs, label.squeeze(1))
                L = L + loss.item()
        val_loss = L / (i + 1)
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred-y_val) ** 2).mean())
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]

        torch.cuda.empty_cache()

        # Test
        if args.test_ratio > 0 and not args.notest_during_training:
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            with torch.no_grad():
                for i, (features,fixedframe,fixthres,label,actlen) in enumerate(test_loader):
                    y_test[i] = scale *label.item()  #
                    features = features.to(device).float()
                    label = label.to(device).float()
                    actlen = actlen.to(device).float()
                    fixthres = fixthres.to(device).float()
                    complexity = fixedframe.to(device).float()
                    outputs = model(features, complexity,actlen,fixthres)
                    y_pred[i] = scale *outputs.item()
                    loss = criterion1(outputs, label.squeeze(1))
                    L = L + loss.item()
            test_loss = L / (i + 1)
            PLCC = stats.pearsonr(y_pred, y_test)[0]
            SROCC = stats.spearmanr(y_pred, y_test)[0]
            RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
            KROCC = stats.stats.kendalltau(y_pred, y_test)[0]

            torch.cuda.empty_cache()

        if not args.disable_visualization:  # record training curves
            writer.add_scalar("loss/train", train_loss, epoch)  #
            writer.add_scalar("loss/val", val_loss, epoch)  #
            writer.add_scalar("SROCC/val", val_SROCC, epoch)  #
            writer.add_scalar("KROCC/val", val_KROCC, epoch)  #
            writer.add_scalar("PLCC/val", val_PLCC, epoch)  #
            writer.add_scalar("RMSE/val", val_RMSE, epoch)  #
            if args.test_ratio > 0 and not args.notest_during_training:
                writer.add_scalar("loss/test", test_loss, epoch)  #
                writer.add_scalar("SROCC/test", SROCC, epoch)  #
                writer.add_scalar("KROCC/test", KROCC, epoch)  #
                writer.add_scalar("PLCC/test", PLCC, epoch)  #
                writer.add_scalar("RMSE/test", RMSE, epoch)  #
        print("EXP ID={}: Update in epoch {}".format(args.exp_id, epoch))
        print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
        #print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
         #     .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        # Update the model with the best val_SROCC
        if val_SROCC > best_val_criterion:
            print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if args.test_ratio > 0 and not args.notest_during_training:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # update best val SROCC

