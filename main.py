import argparse
from utils import *
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import *
from models.Desensitized import get_contrastive_emb_dg
from models.DDA import DDA
from RefineRate import DeleteAdversarialRate
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024,  help='seed')
parser.add_argument("--log", action='store_true',default=False, help='run prepare_data or not')
parser.add_argument('--attack', type=str, default='meta',  help='attack method')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.15,  help='pertubation rate')

parser.add_argument('--recover_percent', type=float, default=0.2,  help='recovery rate')
parser.add_argument('--delta', type=int, default=5,  help='threshold for augmentation')

parser.add_argument('--vanillaGNN', type=str, default='gcn',  help='vanilla GNN')
# citeseer 0.15
parser.add_argument('--k', type=int, default=3,  help='add k neighbors')
parser.add_argument('--jt', type=float, default=0.01,  help='jaccard threshold')
parser.add_argument('--cos', type=float, default=0.1,  help='cosine similarity threshold')
#
parser.add_argument('--alpha', type=float, default=0.95,  help='parameter for graph fusion between GAE graph and original graph')
parser.add_argument('--lamb', type=float, default=3.0,  help='parameter for EP(edge_predict)loss')
parser.add_argument('--temperature', type=float, default=0.45,  help='temperature for GAE')
parser.add_argument('--pretrain_ep', type=int, default=240,  help='epoch of GAE edge prediction pretraining')
parser.add_argument('--pretrain_nc', type=int, default=60,  help='epoch of GNN node classification pretraining')
parser.add_argument('--n_layer', type=int, default=1,  help='layer of gnn')
parser.add_argument('--lr', type=float, default=0.005,  help='learning rate for gnn')
parser.add_argument('--epoch', type=int, default=200,  help='epoch for gnn')


logger=None
args = parser.parse_args()
if args.log:
    logger = get_logger(
        './log/' + args.attack + '/' + 'ours_' + args.dataset + '_' + str(args.ptb_rate) + '.log')
else:
    logger = get_logger('./log/try.log')

if args.attack == 'nettack':
    args.ptb_rate = int(args.ptb_rate)
seed = args.seed
logger.info('seed:{}'.format(seed))
# seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root = './ptb_graphs/'
# Loading data
data = Dataset(root=root, name=args.dataset, setting='prognn', seed=seed)
adj, features, labels = data.adj, data.features, data.labels
if args.dataset == 'pubmed':
    features = features.tocsr()
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
tvt_nids = [idx_train, idx_val, idx_test]
n_nodes = features.shape[0]
n_class = labels.max() + 1
if args.ptb_rate == 0:
    perturbed_adj = adj
    if args.attack == 'nettack':
        idx_test = get_target_nodes(root + args.attack + '/', args.dataset)
elif args.attack in ['meta', 'nettack']:
    perturbed_data = PrePtbDataset(root=root + args.attack + '/',
                                   name=args.dataset,
                                   attack_method=args.attack,
                                   ptb_rate=args.ptb_rate)
    perturbed_adj = perturbed_data.adj
    if args.attack == 'nettack':
        idx_test = perturbed_data.target_nodes
elif args.attack == 'DICE':
    perturbed_adj=np.load(root+'%s/%s_%s_%s.npy' % (args.attack, args.attack, args.dataset, args.ptb_rate),allow_pickle= True).item()
elif args.attack == 'random':
    from deeprobust.graph.global_attack import Random

    attacker = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    attacker.attack(adj, n_perturbations, type='add')
    perturbed_adj = attacker.modified_adj
else:
    perturbed_adj = adj

logger.info('train nodes:%d' % idx_train.shape[0])
logger.info('val nodes:%d' % idx_val.shape[0])
logger.info('test nodes:%d' % idx_test.shape[0])

if __name__ == '__main__':
    logger.info(args)
    perturbed_adj_sparse = perturbed_adj
    logger.info('===start preprocessing the graph===')
    if args.dataset == 'polblogs':
        args.jt = 0
    adj_pre = preprocess_adj(features, perturbed_adj_sparse, logger, threshold=args.jt)
    adj_delete = perturbed_adj_sparse - adj_pre
    logger.info('pre delete edge:%d' % adj_delete.shape[0])
    _, features = to_tensor(perturbed_adj_sparse, features)
    logger.info('===start getting contrastive embeddings===')
    edge_index=to_edge_index(perturbed_adj).to(device)
    delete_edge_index = to_edge_index(adj_delete).to(device)
    pr_weight = pr_drop_weights(edge_index)
    degree_weight = degree_drop_weights(edge_index)

    indices = torch.stack([edge_index[0], edge_index[1]], dim=1)
    mask = (indices[:, 0].unsqueeze(1) == delete_edge_index[0].clone().detach()) & \
           (indices[:, 1].unsqueeze(1) == delete_edge_index[1].clone().detach())
    mask = torch.any(mask, dim=1)
    delete_pr_weight = pr_weight[mask]
    delete_degree_weight = degree_weight[mask]
    embeds, _ = get_contrastive_emb_dg(logger, adj_pre, features.unsqueeze(dim=0).to_dense(),
                                       adj_delete=delete_edge_index,
                                       lr=0.001, weight_decay=0.0, nb_epochs=10000,
                                       delete_pr_weight=delete_pr_weight,
                                       delete_degree_weight=delete_degree_weight,
                                       recover_percent=args.recover_percent
                                       )
    
    embeds = embeds.squeeze(dim=0).to('cpu')
    embeds_matrix = to_scipy(embeds)
    # prune the perturbed graph by the representations
    adj_clean = preprocess_adj(embeds_matrix, perturbed_adj_sparse, logger, jaccard=False, threshold=args.cos)
    adj_clean = sparse_mx_to_sparse_tensor(adj_clean)
    adj_clean = adj_clean.to_dense()
    adj_clean = adj_clean.to(device)
    features = features.to_dense()
    logger.info('===train ours on perturbed graph===')

    acc_total = []
    # add k new neighbors to each node
    get_reliable_neighbors(adj_clean, embeds, k=args.k)
    adj_temp = adj_clean
    # calculate the refinement of adversarial edges
    DeleteAdversarialRate(perturbed_adj, adj_temp, args.dataset)
    
    sparse_matrix = csr_matrix(adj_temp.cpu().numpy())
    del adj,adj_clean,adj_delete,adj_temp,data,degree_weight,delete_edge_index,delete_pr_weight,edge_index,embeds,embeds_matrix,perturbed_adj,perturbed_adj_sparse,pr_weight
    import gc
    gc.collect()
    model = DDA(sparse_matrix, features, labels, tvt_nids, cuda=0, gae=True, alpha=args.alpha,
                beta=args.lamb, temperature=args.temperature, warmup=0, gnnlayer_type=args.vanillaGNN, jknet=False,
                lr=args.lr, n_layers=args.n_layer, log=False, feat_norm='row',
                epochs=args.epoch, delta=args.delta, augment=True)
    acc = model.fit(pretrain_ep=args.pretrain_ep, pretrain_nc=args.pretrain_nc)
    logger.info(acc)






