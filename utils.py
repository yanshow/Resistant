from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import numpy as np
import logging
import torch
from torch_scatter import scatter
from torch_geometric.utils import degree, to_undirected
def adverarial_pr_weights(edge_index, adversarial_edge_index,aggr: str = 'mean', k: int = 10):
    # pv是每个节点的重要性
    pv = compute_pagerank(edge_index, k=k)
    # 如果需要计算 扰动边的度 这里的edge——index相同吗？
    # 可以的话 找到对应扰动边的edge——index 它的起始 终止节点 然后计算扰动边对应的重要性
    # 将这个重要性进行
    pv_row = pv[adversarial_edge_index[0]].to(torch.float32)
    pv_col = pv[adversarial_edge_index[1]].to(torch.float32)
    # 得到边的出节点和入节点的中心权重
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    # mean = 1
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights

def adverarial_degree_weights(edge_index, adversarial_edge_index,aggr: str = 'mean', k: int = 10):
    # pv是每个节点的重要性
    edge_index_ = to_undirected(edge_index)

    deg = degree(edge_index_[1])
    deg_col = deg[adversarial_edge_index[1]].to(torch.float32)
    # deg_row = deg[edge_index[0]].to(torch.float32)
    # 节点中心性（比如采用节点度时）的数值可能跨越 多个数量级，因此设置log来缓解连接密集的节点的影响。
    s_col = torch.log(deg_col)
    # 标准化的过程来获得：
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    # 返回的是每个边的权重
    return weights

def AdversarialEdgeByPagerank(edge_index,adversarial_edge_index,k=10):
    weight = adverarial_pr_weights(edge_index,adversarial_edge_index)
    score, idx = torch.sort(weight, descending=False)
    return score, idx;

def AdversarialEdgeByDegree(edge_index,adversarial_edge_index,k=10):
    weight = adverarial_degree_weights(edge_index,adversarial_edge_index)
    score, idx = torch.sort(weight, descending=False)
    return score, idx;

def compute_pagerank(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x

def compute_degree(edge_index):
    edge_index_ = to_undirected(edge_index)

    deg = degree(edge_index_[1]).to(torch.float32)
    return deg

def extractDegreeNode(edge_index,k= 10):
    degree=compute_pagerank(edge_index,k=k)
    # degree = compute_degree(edge_index)
    score, idx = torch.sort(degree, descending=False)
    return score,idx;
def SelectedNodeID(node_id,node_num,scope):
    start_idx = int(node_num * scope[0])
    end_idx = int(node_num * scope[1])
    idx = node_id[start_idx:end_idx]
    return idx;

def degree_drop_weights(edge_index):
    # edge_index 是一个大小为 2 x E 的整数张量，其中 E 是图中边的数量，每一列表示图中一条边连接的两个节点的索引。
    # 变成无向图
    edge_index_ = to_undirected(edge_index)

    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    # deg_row = deg[edge_index[0]].to(torch.float32)
    # 节点中心性（比如采用节点度时）的数值可能跨越 多个数量级，因此设置log来缓解连接密集的节点的影响。
    s_col = torch.log(deg_col)
    # 标准化的过程来获得：
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    # 返回的是每个边的权重
    return weights


def pr_drop_weights(edge_index, aggr: str = 'mean', k: int = 10):
    pv = compute_pagerank(edge_index, k=k)
    # pv是每个节点的度
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    # 得到边的出节点和入节点的中心权重
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights

def drop_edge_weighted(edge_index, edge_weights, p: float=0.2, threshold: float = 1):
    # 标准化过程p ，threshold
    # torch.manual_seed(6)
    edge_weights = edge_weights / edge_weights.mean() * p
    # 如果edge_weight小于阈值 就把他设为1*阈值
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    # 概率分布丢弃 ！ 有概率
    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]

def SortedCenterEdge(edge_index,k=10):
    weight = pr_drop_weights(edge_index)
    # weight=degree_drop_weights(edge_index)
    score, idx = torch.sort(weight, descending=False)
    return score, idx;

def get_target_nodes(r, dataset):
    import os.path as osp
    import json
    """Get target nodes incides, which is the nodes with degree > 10 in the test set."""
    url = 'https://raw.githubusercontent.com/ChandlerBang/Pro-GNN/master/nettack/{}_nettacked_nodes.json'.format(dataset)
    json_file = osp.join(r,'{}_nettacked_nodes.json'.format(dataset))
    if not osp.exists(json_file):
        print('Dowloading from {} to {}'.format(url, json_file))
        try:
            urllib.request.urlretrieve(url, json_file)
        except:
            raise Exception("Download failed! Make sure you have \
                            stable Internet connection and enter the right name")
    with open(json_file, 'r') as f:
        idx = json.loads(f.read())
    return idx["attacked_test_nodes"]


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


def adj_norm(adj, neighbor_only=False):
    if not neighbor_only:
        adj = torch.add(torch.eye(adj.shape[0]).cuda(), adj)
    if adj.is_sparse:
        degree = adj.to_dense().sum(dim=1)
    else:
        degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), -0.5).expand(adj.shape[0], adj.shape[0])
    in_degree_norm = torch.where(torch.isinf(in_degree_norm), torch.full_like(in_degree_norm, 0), in_degree_norm)
    out_degree_norm = torch.pow(degree.view(-1, 1), -0.5).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.where(torch.isinf(out_degree_norm), torch.full_like(out_degree_norm, 0), out_degree_norm)
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    return adj


def sparse_dense_mul(s, d):
    if not s.is_sparse:
        return s * d
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def evaluate(model, adj, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(adj, features)
        logits = logits[mask]
        test_labels = labels[mask]
        _, indices = logits.max(dim=1)
        correct = torch.sum(indices == test_labels)
        return correct.item() * 1.0 / test_labels.shape[0]


def get_reliable_neighbors(adj, features, k, degree_threshold=1):
    degree = adj.sum(dim=1)
    degree_mask = degree > degree_threshold
    assert degree_mask.sum().item() >= k
    sim = cosine_similarity(features)
    sim = torch.FloatTensor(sim).to('cuda')
    sim[:, degree_mask == False] = 0
    _, top_k_indices = sim.topk(k=k, dim=1)
    for i in range(adj.shape[0]):
        adj[i][top_k_indices[i]] = 1
        adj[i][i] = 0
    return


def adj_new_norm(adj, alpha):
    adj = torch.add(torch.eye(adj.shape[0]).cuda(), adj)
    degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), alpha).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.pow(degree.view(-1, 1), alpha).expand(adj.shape[0], adj.shape[0])
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    if alpha != -0.5:
        return adj / (adj.sum(dim=1).reshape(adj.shape[0], -1))
    else:
        return adj


def preprocess_adj(features, adj, logger, metric='similarity', threshold=0.03, jaccard=True):
    """Drop dissimilar edges.(Faster version using numba)
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    adj_triu = sp.triu(adj, format='csr')

    if sp.issparse(features):
        features = features.todense().A  # make it easier for njit processing

    if metric == 'distance':
        removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    else:
        if jaccard:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                           threshold=threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                          threshold=threshold)
    logger.info('removed %s edges in the original graph' % removed_cnt)
    modified_adj = adj_triu + adj_triu.transpose()
    return modified_adj


def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


def dropedge_both(A, iA, jA, features, threshold1=2.5, threshold2=0.01):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C1 = np.linalg.norm(features[n1] - features[n2])

            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C2 = inner_product / (np.sqrt(np.square(a).sum() + np.square(b).sum())+ 1e-6)
            if C1 > threshold1 or threshold2 < 0:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a*b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)
            if C <= threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def sparse_mx_to_sparse_tensor(sparse_mx):
    """sparse matrix to sparse tensor matrix(torch)
    Args:
        sparse_mx : scipy.sparse.csr_matrix
            sparse matrix
    """
    sparse_mx_coo = sparse_mx.tocoo().astype(np.float32)
    sparse_row = torch.LongTensor(sparse_mx_coo.row).unsqueeze(1)
    sparse_col = torch.LongTensor(sparse_mx_coo.col).unsqueeze(1)
    sparse_indices = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparse_indices.t(), sparse_data, torch.Size(sparse_mx.shape))


def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor on target device.
    Args:
        adj : scipy.sparse.csr_matrix
            the adjacency matrix.
        features : scipy.sparse.csr_matrix
            node features
        labels : numpy.array
            node labels
        device : str
            'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)


def idx_to_mask(idx, nodes_num):
    """Convert a indices array to a tensor mask matrix
    Args:
        idx : numpy.array
            indices of nodes set
        nodes_num: int
            number of nodes
    """
    mask = torch.zeros(nodes_num)
    mask[idx] = 1
    return mask.bool()


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.
    Args:
        tensor : torch.Tensor
                 given tensor
    Returns:
        bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def to_edge_index(csr_adj):
    """Convert a tensor to edge index"""
    coo_adj = sp.coo_matrix(csr_adj)
    edge_index = torch.tensor([np.array(coo_adj.row), np.array(coo_adj.col)])
    return edge_index.to(torch.int64)
