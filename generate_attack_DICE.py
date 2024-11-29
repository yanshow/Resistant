from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from scipy import sparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=int(time.time()), help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora_ml', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
parser.add_argument('--model', type=str, default='DICE',help='attack method')

parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# datasets=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed']
# ptb_rates=[0.05,0.1,0.15,0.2]
# for dataset in datasets:
#     args.dataset=dataset
#     for ptb_rate in ptb_rates:
#         args.ptb_rate = ptb_rate
data = Dataset(root='ptb_graphs', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
perturbations = int(args.ptb_rate * (adj.sum()//2))
adj_tensor, features_tensor, labels_tensor = preprocess(adj, features, labels, preprocess_adj=False)

model = DICE()
model = model.to(device)

def test(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features_tensor.shape[1],
              nhid=args.hidden,
              nclass=labels_tensor.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    # gcn.fit(features, adj, labels, idx_train) # train without model picking
    gcn.fit(features_tensor, adj, labels_tensor, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels_tensor[idx_test])
    acc_test = accuracy(output[idx_test], labels_tensor[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    print('=== testing GCN on original(clean) graph ===')
    test(adj_tensor)
    model.attack(adj, labels, n_perturbations=perturbations)
    modified_adj = model.modified_adj.tocsr()
    np.save("./ptb_graphs/%s/%s_%s_%s" % ('DICE', 'DICE', args.dataset, args.ptb_rate), modified_adj)
    print('=== testing GCN on perturbed graph ===')
    test(modified_adj)

if __name__ == '__main__':
    main()