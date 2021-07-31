# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN

BUIR implementation
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models.common.abstract_recommender import GeneralRecommender
from models.common.loss import BPRLoss, EmbLoss, SSLLoss
from models.common.init import xavier_uniform_initialization

# from data import ppr


class Diff_Attn(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    def __init__(self, config, dataset):
        super(Diff_Attn, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.embedding_dict = self._init_model()

        # generate normalized adj matrix
        self.adj = self.get_adj_mat()
        # sp.save_npz('data/' + config['dataset'] + '_adj.npz', self.adj.tocsr())
        self.norm_adj_matrix = self.sparse_mat_to_tensor(self.get_norm_adj_mat(self.adj)).to(self.device)

        # generate ppr matrix
        ppr_mat = self.get_ppr(config['dataset'], config['alpha_u'], config['alpha_i'], config['eps'], config['topu'], config['topi'], config['ppr_norm'])
        self.ppr_mat = self.sparse_mat_to_tensor(ppr_mat.tocoo()).to(self.device)

        self.diff_users = self.sparse_mat_to_tensor(ppr_mat[:self.n_users, :self.n_users].tocoo()).to(self.device)
        self.diff_items = self.sparse_mat_to_tensor(ppr_mat[self.n_users:, self.n_users:].tocoo()).to(self.device)
        self.diff_ui = self.sparse_mat_to_tensor(
            self.get_diff_ui_mat(ppr_mat, config['dataset'], config['alpha'], config['eps'], config['k'],
                                 config['ppr_norm'])).to(self.device)

        # self.ssl_loss = SSLLoss(config['ssl_temp'])
        # self.reg_ssl = config['reg_ssl']

        # parameters initialization
        #self.apply(xavier_uniform_initialization)

        self.config = config
        if config['method'] == 0:
            self.attn_users = Attention(self.latent_dim, 0.1)
            self.attn_items = Attention(self.latent_dim, 0.1)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.latent_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.latent_dim)))
        })

        return embedding_dict

    def get_diff_ui_mat(self, mat, ds, alpha, eps, topk, ppr_norm):
        fname = f'{ds}_s_alpha_{str(alpha)}_eps_{str(eps)}_top{str(topk)}_{str(ppr_norm)}_sep_diff_ui.npz'
        try:
            mat = sp.load_npz('data/' + fname)
        except:
            mat = mat.tolil()
            mat[:self.n_users, :self.n_users] = 0
            mat[self.n_users:, self.n_users:] = 0
            sp.save_npz('data/' + fname, mat.tocsr())
        return mat.tocoo()

    def get_adj_mat(self):
        """
        Get the interaction matrix of users and items.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        return A

    def get_norm_adj_mat(self, A):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return L

    def get_ppr(self, ds, alpha_u, alpha_i, eps, topu, topi, ppr_norm):
        fname = f'{ds}_s_alpha_{str(alpha_u)}_{str(alpha_i)}_eps_{str(eps)}_topu_{str(topu)}_topi_{str(topi)}_{str(ppr_norm)}_sep.npz'
        # fname = f'{ds}_s_alpha_{str(alpha)}_eps_{str(eps)}_top{str(topk)}_{str(ppr_norm)}_test.npz'
        try:
            topk_mat = sp.load_npz('data/' + fname)
        except:
            train_idx = np.array([i for i in range(int(self.n_users + self.n_items))])
            topk_mat = ppr.topk_ppr_matrix(self.n_users, self.n_items, self.adj.tocsr(), alpha_u, alpha_i, eps, train_idx, topu, topi,
                                           normalization=ppr_norm)
            sp.save_npz('data/' + fname, topk_mat)
        return topk_mat

    def sparse_mat_to_tensor(self, L):
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        # user_embeddings = self.user_embedding.weight
        # item_embeddings = self.item_embedding.weight
        # ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        return ego_embeddings

    def forward(self):
        # all_embeddings = self.get_ego_embeddings()
        all_embeddings_ppr = self.get_ego_embeddings()

        user_emb_ppr = all_embeddings_ppr[:self.n_users, :]
        user_emb_ppr = torch.sparse.mm(self.diff_users, user_emb_ppr)

        item_emb_ppr = all_embeddings_ppr[self.n_users:, :]
        item_emb_ppr = torch.sparse.mm(self.diff_items, item_emb_ppr)

        all_emb_ppr = torch.sparse.mm(self.diff_ui, all_embeddings_ppr)
        all_emb_ppr_users, all_emb_ppr_items = torch.split(all_emb_ppr, [self.n_users, self.n_items])

        # user_ppr_all_embeddings = self.attn_users([user_emb_ppr, all_emb_ppr_users])
        # item_ppr_all_embeddings = self.attn_items([item_emb_ppr, all_emb_ppr_items])

        if self.config['method'] == 1:
            user_ppr_all_embeddings = torch.cat((user_emb_ppr, all_emb_ppr_users), dim=1)
            item_ppr_all_embeddings = torch.cat((item_emb_ppr, all_emb_ppr_items), dim=1)
        elif self.config['method'] == 2:
            user_ppr_all_embeddings = torch.mean(torch.stack([user_emb_ppr, all_emb_ppr_users], dim=1), dim=1)
            item_ppr_all_embeddings = torch.mean(torch.stack([item_emb_ppr, all_emb_ppr_items], dim=1), dim=1)
        elif self.config['method'] == 3:
            user_ppr_all_embeddings = user_emb_ppr + all_emb_ppr_users
            item_ppr_all_embeddings = item_emb_ppr + all_emb_ppr_items
        elif self.config['method'] == 0:
            user_ppr_all_embeddings = self.attn_users([user_emb_ppr, all_emb_ppr_users])
            item_ppr_all_embeddings = self.attn_items([item_emb_ppr, all_emb_ppr_items])

        return user_ppr_all_embeddings, item_ppr_all_embeddings

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user, :]
        posi_embeddings = item_all_embeddings[pos_item, :]
        negi_embeddings = item_all_embeddings[neg_item, :]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.embedding_dict['user_emb'][user, :]
        posi_ego_embeddings = self.embedding_dict['item_emb'][pos_item, :]
        negi_ego_embeddings = self.embedding_dict['item_emb'][neg_item, :]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        # calculate ssl loss
        # ssl_loss = self.ssl_loss(user_all_embeddings[user, :], user_ppr_all_embeddings[user, :],
        #                          item_all_embeddings[pos_item, :], item_ppr_all_embeddings[pos_item, :])
        #
        # loss = mf_loss + self.reg_weight * reg_loss + self.reg_ssl * ssl_loss
        # print(mf_loss, self.reg_weight * reg_loss, ssl_loss)
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        # use diff matrix for prediction
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user, :]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))

        # # use adj matrix for prediction
        # all_embeddings = self.get_ego_embeddings()
        #
        # all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
        # u_embeddings = all_embeddings[:self.n_users, :][user, :]
        # i_embeddings = all_embeddings[self.n_users:, :]
        #
        # # dot with all item embedding to accelerate
        # scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp