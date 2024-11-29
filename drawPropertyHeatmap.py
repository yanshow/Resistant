import time
import argparse
# datasets
from deeprobust.graph.data import Dataset, PrePtbDataset
# arguments
import os
from utils import *
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import accuracy
from torch_geometric.utils import to_dense_adj
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from RefineRate import adversarialEdge
from utils import AdversarialEdgeByPagerank, AdversarialEdgeByDegree

from RefineRate import EdgeDiffPositive
def adversarialEdge(perturbG_dense,dataset):
    data = Dataset(root='./ptb_graphs/', name=dataset, setting='prognn')
    originG = data.adj.todense()
    perturbG = perturbG_dense.todense()
    AdversarialEdge = EdgeDiffPositive(perturbG, originG)
    return AdversarialEdge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.25, help='pertubation rate')
    parser.add_argument('--attack', type=str, default='meta', help='attack method')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = './ptb_graphs/'
    datasetList=['citeseer']
    # datasetList = ['cora_ml','cora', 'citeseer', 'polblogs', 'pubmed']
    # measure_method='PageRank'
    measure_method = 'Degree'
    perturbrateList = [0.05,0.1, 0.15, 0.2, 0.25]
    for dataset in datasetList:
        result = []
        for perturbrate in perturbrateList:
            args.dataset=dataset
            args.ptb_rate=perturbrate
            setC = 5
            scope = [[i / setC, (i + 1) / setC] for i in range(0, setC)]
            acc_all=[]
            for scope_temp in scope:
                data = Dataset(root=root, name=args.dataset, setting='prognn', seed=seed)
                adj, features, labels = data.adj, data.features, data.labels
                if args.dataset == 'pubmed':
                    # pumbmed 要变密集特征
                    features = features.tocsr()
                idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
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
                elif args.attack == 'random':
                    from deeprobust.graph.global_attack import Random
                    attacker = Random()
                    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
                    attacker.attack(adj, n_perturbations, type='add')
                    perturbed_adj = attacker.modified_adj
                else:
                    perturbed_adj = adj

                # from label indice to mask tensor
                train_mask, val_mask, test_mask = idx_to_mask(idx_train, n_nodes), idx_to_mask(idx_val, n_nodes), \
                    idx_to_mask(idx_test, n_nodes)
                train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
                # the disturbed graph becomes dense and becomes a float format
                perturbed_adj = torch.FloatTensor(perturbed_adj.todense()).to(device)
                perturbed_adj_sparse = to_scipy(perturbed_adj)

                # convert PyTorch tensor to Scipy sparse matrix
                perturbed_sparse = sp.coo_matrix(perturbed_adj.cpu().numpy())
                edge_index = torch.tensor([perturbed_sparse.row, perturbed_sparse.col]).to(device)
                edge_index=edge_index.to(torch.int64)
                # export the adversarial edge and turn it into the edge_index format
                AdversarialEdgeGraph = adversarialEdge(perturbed_adj_sparse, args.dataset)
                adversarial_sparse=sp.coo_matrix(AdversarialEdgeGraph)
                adversarial_edge_index = torch.tensor([adversarial_sparse.row, adversarial_sparse.col]).to(device)
                adversarial_edge_index = adversarial_edge_index.to(torch.int64)
                # compute the rank of adversarial edges
                if measure_method=='PageRank':
                    AdvEdgeScore,AdvEdgeID=AdversarialEdgeByPagerank(edge_index,adversarial_edge_index)
                else:
                    AdvEdgeScore,AdvEdgeID=AdversarialEdgeByDegree(edge_index,adversarial_edge_index)
                # the id of adversarial edges then inject adversarial edges at related percent into origin graph
                acc = []
                times = 10
                for run in range(times):
                    # get related edges
                    total_elements = adversarial_edge_index.shape[1]
                    start_idx = int(total_elements * scope_temp[0])
                    end_idx = int(total_elements * scope_temp[1])
                    # select edgeid in the scope
                    idx = AdvEdgeID[start_idx:end_idx]
                    # get the selected adversarial edges
                    SelectedEdges = adversarial_edge_index[:,idx]
                    adver_adj = to_dense_adj(SelectedEdges, max_num_nodes=n_nodes).squeeze(0).cpu().numpy()
                    # inject adversarial edges(adver_adj) into origin graph(origin_adj)
                    final_adj = adj.todense()+adver_adj
                    # put the mixed graph into gcn training to see the effect of the importance of different adversarial edges
                    model = GCN(nfeat=features.shape[1],
                                nhid=64,
                                nclass=labels.max().item() + 1, device=device, lr=0.001)
                    model.fit(features, final_adj, labels, idx_train, train_iters=500, idx_val=idx_val)
                    output = model.output.cpu()
                    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
                    acc_test = accuracy(output[idx_test], labels[idx_test]).item() * 100
                    acc.append(acc_test)
                print('{:.2f}±{:.2f}'.format(np.mean(acc), np.std(acc)))
                avg_acc = np.round(np.mean(acc), 2)
                std_dev = np.round(np.std(acc), 2)
                acc_all.append(avg_acc)
            print('acc_all',acc_all)
            result.append(acc_all)
        print(result)

        data = result

        f, ax = plt.subplots(figsize=(10, 8))
        ax = sns.heatmap(data, annot=True, fmt='.2f', cmap='RdBu', annot_kws={"size": 18})
        # plt.tight_layout()  # Set a compact layout
        # Adjust layout
        plt.subplots_adjust(left=0.1, right=1, top=0.95, bottom=0.15)
        # Set title
        plt.title('{} ({})'.format(measure_method, dataset), fontsize=18)
        # plt.title('{}'.format(dataset),fontsize=18)
        # Set the horizontal and vertical headings

        ax.set_xlabel('Edge Influence (%)', fontsize=18)  # Set the X-axis label font size
        ax.set_ylabel('Perturb Rate (%)', fontsize=18)  # Set the Y-axis label font size
        # Set specific values for the horizontal and vertical coordinates
        ax.set_xticklabels(['0~20', '20~40', '40~60', '60~80', '80~100'], rotation=45)
        ax.set_yticklabels(['5', '10', '15', '20','25'], rotation=0)  # Add percent

        # Display a numerical heat map
        image_path = './log'
        image_name = '{}_{}.pdf'.format(measure_method,dataset)
        image_png = '{}_{}.png'.format(measure_method,dataset)
        plt.savefig(os.path.join(image_path, image_name))
        plt.savefig(os.path.join(image_path, image_png))
        plt.show()



