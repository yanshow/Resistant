import argparse
from utils import *
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import *
from models.Desensitized import get_contrastive_emb_dg
from models.DDA import DDA
from RefineRate import DeleteAdversarialRate
from scipy.sparse import csr_matrix
import os
import optuna
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024, help='seed')
parser.add_argument("--log", action='store_true', default=False, help='run prepare_data or not')
parser.add_argument('--attack', type=str, default='meta', help='attack method')
parser.add_argument('--datasets', nargs='+', default=['cora'], help='datasets')
parser.add_argument('--ptb_rates', nargs='+', type=float, default=[0.05,0.1,0.15,0.2], help='perturbation rates')
parser.add_argument('--vanillaGNN', type=str, default='gsage', help='vanilla GNN')
args = parser.parse_args()

def main(trial, dataset, ptb_rate):
    if args.log:
        logger = get_logger('./log/' + args.attack + '/' + 'ours_' + dataset + '_' + str(ptb_rate) + '.log')
    else:
        logger = get_logger('./log/try.log')

    if args.attack == 'nettack':
        ptb_rate = int(ptb_rate)
    seed = args.seed
    logger.info('seed:{}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = './ptb_graphs/'
    # Loading data
    data = Dataset(root=root, name=dataset, setting='prognn', seed=seed)
    adj, features, labels = data.adj, data.features, data.labels
    if dataset == 'pubmed':
        features = features.tocsr()
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    tvt_nids = [idx_train, idx_val, idx_test]
    n_nodes = features.shape[0]
    n_class = labels.max() + 1

    # Load perturbed adjacency matrix
    if ptb_rate == 0:
        perturbed_adj = adj
        if args.attack == 'nettack':
            idx_test = get_target_nodes(root + args.attack + '/', dataset)
    elif args.attack in ['meta', 'nettack']:
        perturbed_data = PrePtbDataset(root=root + args.attack + '/', name=dataset, attack_method=args.attack, ptb_rate=ptb_rate)
        perturbed_adj = perturbed_data.adj
        if args.attack == 'nettack':
            idx_test = perturbed_data.target_nodes
    elif args.attack == 'DICE':
        perturbed_adj=np.load(root+'%s/%s_%s_%s.npy' % (args.attack, args.attack, dataset, ptb_rate),allow_pickle= True).item()
    elif args.attack == 'random':
        from deeprobust.graph.global_attack import Random
        attacker = Random()
        n_perturbations = int(ptb_rate * (adj.sum() // 2))
        attacker.attack(adj, n_perturbations, type='add')
        perturbed_adj = attacker.modified_adj
    else:
        perturbed_adj = adj

    logger.info('train nodes:%d' % idx_train.shape[0])
    logger.info('val nodes:%d' % idx_val.shape[0])
    logger.info('test nodes:%d' % idx_test.shape[0])

    # Start Optuna optimization
    logger = get_logger('./log/optuna/' + dataset + '_' + str(ptb_rate) + '_' + time.strftime('_%m%d-%H%M%S') + '.log')
    params = {
        'k': trial.suggest_int("k", 0, 20, step=1),
        'jt': trial.suggest_float("jt", 0, 0.05, step=0.01),
        'cos': trial.suggest_float("cos", 0, 0.3, step=0.1),
        'temperature': trial.suggest_float("temperature", 0, 0.95, step=0.05),
        'alpha': trial.suggest_float("alpha", 0, 0.95, step=0.05),
        'lamb': trial.suggest_float("lamb", 0, 3, step=0.05),
        'lr': trial.suggest_float("lr", 0.001, 0.01, step=0.001),
        'n_layer': trial.suggest_int("n_layer", 1, 3, step=1),
        'epochs': trial.suggest_int("epochs", 200, 500, step=50),
        'pretrain_ep': trial.suggest_int("pretrain_ep", 10, 250, step=10),
        'pretrain_nc': trial.suggest_int("pretrain_nc", 10, 250, step=10),
        'recover_percent': trial.suggest_float("recover_percent", 0.1, 0.5, step=0.1),
        'delta': trial.suggest_int("delta", 5, 20, step=5)
    }

    # Set parameters
    args.k = params['k']
    args.jt = params['jt']
    args.cos = params['cos']
    args.temperature = round(params['temperature'],2)
    args.alpha = round(params['alpha'],2)
    args.lamb = round(params['lamb'],2)
    args.lr = params['lr']
    args.n_layer = params['n_layer']
    args.epoch = params['epochs']
    args.pretrain_ep = params['pretrain_ep']
    args.pretrain_nc = params['pretrain_nc']
    args.recover_percent=0.2
    args.delta=5
    # args.recover_percent = round(params['recover_percent'],2)
    # args.delta = params['delta']

    logger.info(args)
    perturbed_adj_sparse = perturbed_adj
    logger.info('===start preprocessing the graph===')
    if dataset == 'polblogs':
        args.jt = 0
    adj_pre = preprocess_adj(features, perturbed_adj_sparse, logger, threshold=args.jt)
    adj_delete = perturbed_adj_sparse - adj_pre
    logger.info('pre delete edge:%d' % adj_delete.shape[0])
    _, features = to_tensor(perturbed_adj_sparse, features)
    logger.info('===start getting contrastive embeddings===')

    edge_index = to_edge_index(perturbed_adj).to(device)
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
                                       recover_percent=args.recover_percent)

    embeds = embeds.squeeze(dim=0).to('cpu')
    embeds_matrix = to_scipy(embeds)
    adj_clean = preprocess_adj(embeds_matrix, perturbed_adj_sparse, logger, jaccard=False, threshold=args.cos)
    adj_clean = sparse_mx_to_sparse_tensor(adj_clean)
    adj_clean = adj_clean.to_dense()
    adj_clean = adj_clean.to(device)
    features = features.to_dense()
    logger.info('===train ours on perturbed graph===')

    acc_total = []
    get_reliable_neighbors(adj_clean, embeds, k=args.k)
    adj_temp = adj_clean
    DeleteAdversarialRate(perturbed_adj, adj_temp, dataset)

    sparse_matrix = csr_matrix(adj_temp.cpu().numpy())
    del adj, adj_clean, adj_delete, adj_temp, data, degree_weight, delete_edge_index, delete_pr_weight, edge_index, embeds, embeds_matrix, perturbed_adj, perturbed_adj_sparse, pr_weight
    import gc
    gc.collect()

    for run in range(3):
        model = DDA(sparse_matrix, features, labels, tvt_nids, cuda=0, gae=True, alpha=args.alpha,
                    beta=args.lamb, temperature=args.temperature, warmup=0, gnnlayer_type=args.vanillaGNN, jknet=False,
                    lr=args.lr, n_layers=args.n_layer, log=False, feat_norm='row',
                    epochs=args.epoch, delta=args.delta, augment=True)
        acc = model.fit(pretrain_ep=args.pretrain_ep, pretrain_nc=args.pretrain_nc)
        acc_total.append(acc)

    logger.info(acc_total)
    logger.info('Mean Accuracy:%f' % np.mean(acc_total))
    logger.info('Standard Deviation:%f' % np.std(acc_total, ddof=1))
    logger.info('Result:{:.2f}±{:.2f}'.format(round(np.mean(acc_total) * 100, 2), round(np.std(acc_total, ddof=1) * 100, 2)))
    return np.mean(acc_total) * 100

if __name__ == "__main__":
    if not os.path.exists('./log/optuna/'):
        os.makedirs('./log/optuna/')

    for dataset in args.datasets:
        for ptb_rate in args.ptb_rates:
            str_name = args.attack + '-' + dataset + '-' + str(ptb_rate) + time.strftime('_%m%d-%H%M%S') + '-Resistant'
            study = optuna.create_study(study_name=str_name, direction="maximize", sampler=optuna.samplers.TPESampler(), storage='sqlite:///D:/expResult/MyStable/optuna/db.' + str_name)
            study.optimize(lambda trial: main(trial, dataset, ptb_rate), n_trials=1000)

            trial = study.best_trial
            print(f"Dataset: {dataset}, Perturbation Rate: {ptb_rate}")
            print("Best accuracy: : ", trial.value)
            print("Best params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            history = optuna.visualization.plot_optimization_history(study)
            importance = optuna.visualization.plot_param_importances(study)
            optuna.visualization.plot_slice(study)
            parallel = optuna.visualization.plot_parallel_coordinate(study)
