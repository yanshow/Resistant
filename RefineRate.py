from deeprobust.graph.data import Dataset
import numpy as np


def DetectNan(num):
    result = num if num else 0
    return result;
def DeleteAdversarialRate(perturbG,refinedG,dataset):
    data = Dataset(root='./ptb_graphs/', name=dataset, setting='prognn')
    originG = data.adj.todense()
    perturbG=perturbG.todense()
    refinedG=refinedG.detach()
    refinedG=refinedG.cpu().numpy()
    refinedG[refinedG != 0] = 1.0
    # 精炼后消除的边
    DeleteEdge=EdgeDiffPositive(perturbG,refinedG)
    # 对抗边
    AdversarialEdge=EdgeDiffPositive(perturbG,originG)
    # 被消除的对抗边
    DeleteAdversarialEdge = np.where((DeleteEdge == 1) & (AdversarialEdge == 1),1,0)
    # 被消除的对抗边占总消除边的数量
    dividend=DetectNan(np.sum(DeleteEdge==1))
    if dividend!=0:
        deleteAdverRate=DetectNan(np.sum(DeleteAdversarialEdge==1))/DetectNan(np.sum(DeleteEdge==1))
    else:
        deleteAdverRate=0
    print('Rate of Pruning Adversarial Edges :%f' % deleteAdverRate)
    return deleteAdverRate
    # extra_nonzero_indices = np.nonzero(matrix2 - matrix1)
    # result_matrix = np.zeros_like(matrix1)
    # result_matrix[extra_nonzero_indices] = 1
    # extra_nonzero_indices = torch.nonzero(matrix2 - matrix1)
def EdgeDiffPositive(G1,G2):
    indice_matrix = np.where((G1 - G2) == 1, 1, 0)
    return indice_matrix;

def DetectLowDegreeNode(adj,lowerBound=0.0,upperBound=0.1,mask=None):
    row_sums = (np.add.reduce(adj, axis=1) + np.add.reduce(adj, axis=0))/2
    node_degree=np.array(row_sums).flatten()# 计算出每个节点的度
    lowerBoundValue = int(lowerBound * adj.shape[1])  # 找出规定范围内的元素个数
    upperBoundValue = int(upperBound * adj.shape[1])
    sorted_indices = np.argsort(node_degree)
    sorted_indices = sorted_indices[lowerBoundValue:upperBoundValue]
    sorted_values = node_degree[sorted_indices]

    return sorted_indices,sorted_values

def DetectNodeDegree(adj):
    row_sums = (np.add.reduce(adj, axis=1) + np.add.reduce(adj, axis=0))
    node_degree = np.array(row_sums).flatten()  # 计算出每个节点的度
    sorted_indices = np.argsort(node_degree)
    sorted_values = node_degree[sorted_indices]
    return sorted_indices,sorted_values

def DetectLowDegreeEdge3(adj,lowerBound,upperBound):
    node_indices_low,node_values_low=DetectNodeDegree(adj)
    lowDegreeEdge=node_indices_low[(lowerBound<node_values_low)&(node_values_low<=upperBound)]
    row_indexes = lowDegreeEdge
    col_indexes = lowDegreeEdge
    result = np.zeros_like(adj)
    result[row_indexes[:, None], col_indexes] = adj[row_indexes[:, None], col_indexes]
    print('Low Degree Edge number:%f' % (DetectNan(np.sum(result == 1))))
    return result

def DetectLowDegreeEdge2(adj,lowerBound,upperBound):
    from scipy.sparse import coo_matrix
    adj_try=coo_matrix(adj)
    edgeNum = adj_try.row.shape[0]
    row_sums = np.add.reduce(adj, axis=1) + np.add.reduce(adj, axis=0)
    node_degree = np.array(row_sums).flatten()
    edge_degree=np.zeros_like(adj_try.row)
    for i in range(edge_degree.shape[0]):
        edge_degree[i]=node_degree[adj_try.row[i]]+node_degree[adj_try.col[i]]
    # 取出对应id
    lowerBoundValue = int(lowerBound * edgeNum)
    upperBoundValue = int(upperBound * edgeNum)
    sorted_indices=np.argsort(edge_degree)
    sorted_indices = sorted_indices[lowerBoundValue:upperBoundValue]
    sorted_values = edge_degree[sorted_indices]
    row = adj_try.row[sorted_indices]
    col = adj_try.col[sorted_indices]
    data = np.ones(sorted_indices.shape[0])
    lowDegreeEdgeAdj=coo_matrix((data, (row, col)), shape=(adj.shape[0], adj.shape[0])).toarray()
    return lowDegreeEdgeAdj



def DetectLowDegreeEdge(adj,lowerBound,upperBound):
    # 第一种 低度的点之间
    # 两种 一种两个节点的度都比较小 另一种是 边=dv1+dv2度较小
    node_indices_low,node_values_low=DetectLowDegreeNode(adj,lowerBound,upperBound)
    node_indices_up, node_values_up = DetectLowDegreeNode(adj, lowerBound, upperBound)
    print(node_values_low,node_values_up)
    # 创建一个示例的二维数组
    # 指定要保留的行索引和列索引（假设是一维数组）
    row_indexes = node_indices_low  # 要保留的行索引
    col_indexes = node_indices_up  # 要保留的列索引

    result = np.zeros_like(adj)
    result[row_indexes[:, None], col_indexes] = adj[row_indexes[:, None], col_indexes]
    print('Low Degree Edge number:%f' % (DetectNan(np.sum(result==1))))
    return result


def LowDegreeAdverRate(perturbG_dense,dataset,lowerBoundValue=0.0,upperBoundValue=1.0):
    print(lowerBoundValue, upperBoundValue)
    data = Dataset(root='tmp/', name=dataset, setting='nettack')
    originG = data.adj.todense()
    perturbG = perturbG_dense.todense()
    AdversarialEdge = EdgeDiffPositive(perturbG, originG)
    # 对抗边占所有边比例
    advEdgeRate=DetectNan(np.sum(AdversarialEdge==1))/DetectNan(np.sum(perturbG==1))
    # advEdgeRate = np.sum(AdversarialEdge == 1) / np.sum(originG == 1)
    print('Adverial Edge Rate:%f' % advEdgeRate)
    LowDegreeG=DetectLowDegreeEdge2(perturbG,lowerBoundValue,upperBoundValue)
    LowDegreeAdverG = np.where((LowDegreeG == 1) & (AdversarialEdge == 1), 1, 0)
    if DetectNan(np.sum(LowDegreeG==1)!=0):
        LowDegreeAdverialRate=DetectNan(np.sum(LowDegreeAdverG==1))/DetectNan(np.sum(LowDegreeG==1))
        print('Low Degree Adverial Edge Rate:%f' % LowDegreeAdverialRate)
    else:
        LowDegreeAdverialRate=0
    return LowDegreeAdverialRate,advEdgeRate

def LowDegreeAdverRateTest(perturbG_dense,dataset,lowerBoundValue=0.0,upperBoundValue=1.0):
    print(lowerBoundValue, upperBoundValue)
    data = Dataset(root='tmp/', name=dataset, setting='nettack')
    originG = data.adj.todense()
    perturbG = perturbG_dense.todense()
    AdversarialEdge = EdgeDiffPositive(perturbG, originG)
    # 对抗边占所有边比例
    advEdgeRate=DetectNan(np.sum(AdversarialEdge==1))/DetectNan(np.sum(perturbG==1))
    print('Adverial Edge Rate:%f' % advEdgeRate)
    LowDegreeG=DetectLowDegreeEdge3(perturbG,lowerBoundValue,upperBoundValue)
    LowDegreeAdverG = np.where((LowDegreeG == 1) & (AdversarialEdge == 1), 1, 0)
    if DetectNan(np.sum(LowDegreeG==1)!=0):
        LowDegreeAdverialRate=DetectNan(np.sum(LowDegreeAdverG==1))/DetectNan(np.sum(LowDegreeG==1))
        print('Low Degree Adverial Edge Rate:%f' % LowDegreeAdverialRate)
    else:
        LowDegreeAdverialRate=0
    return DetectNan(np.sum(LowDegreeAdverG==1)),DetectNan(np.sum(LowDegreeG==1))-DetectNan(np.sum(LowDegreeAdverG==1))


def adversarialEdge(perturbG_dense,dataset):
    data = Dataset(root='../ptb_graphs/', name=dataset, setting='prognn')
    originG = data.adj.todense()
    perturbG = perturbG_dense.todense()
    AdversarialEdge = EdgeDiffPositive(perturbG, originG)
    return AdversarialEdge
