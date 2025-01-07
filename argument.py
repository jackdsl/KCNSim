import argparse
import torch

def parse_opt():
    xiaoceng = "6_1"
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default="5", help="The random seed")
    parser.add_argument('--dataset', type=str, default="bird_count", help="The dataset name: currently can only be 'bird_count'")
    parser.add_argument('--data_path', type=str, default="./datasets", help="The folder containing the data file. The default file is './data/{dataset}.pkl'")
    parser.add_argument('--use_default_test_set', type=bool, default=True, help='Use the default test set from the data')
    parser.add_argument("--Spatial_data_to_be_predicted", type=str, default=fr'C:\工作数据\KCN数据\{xiaoceng}\predicting_data.npz', help="The data needed to be predicted")
    parser.add_argument("--predict", type=bool, default=True, help="True if want to predict full space")
    parser.add_argument('--Spatial_data_predicted', type=str, default=fr'C:\工作数据\KCN数据\{xiaoceng}\predicted_data.txt', help="The data had been predicted")
    parser.add_argument('--train_file', type=str, default=fr'C:\工作数据\KCN数据\{xiaoceng}\data.npz')

    parser.add_argument('--feature_mean', type=float, help="This is the mean of training features")
    parser.add_argument('--feature_std', type=float, help="This is the std of training features")

    parser.add_argument('--model', type=str, default='kcn', help='One of three model types, kcn, kcn_gat, kcn_sage, which use GCN, GAT, and GraphSAGE respectively')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors')
    parser.add_argument('--length_scale', default="auto", help='Length scale for RBF kernel. If set to "auto", then it will be set to the median of neighbor distances')
    parser.add_argument('--hidden_sizes', type=list, default=[8, 8, 8], help='Number of units in hidden layers, also decide the number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--last_activation', type=str, default='none', help='Activation for the last layer')
    
    parser.add_argument('--loss_type', type=str, default='squared_error', help='Loss type') 
    parser.add_argument('--validation_size', type=int, default=5, help='Validation size')
    
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--es_patience', type=int, default=20, help='Patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    
    parser.add_argument('--device', type=str, default="auto", help='Computation device.')

    parser.add_argument('--coordinate_transform', type=bool, default=False, help='Use coordinate transformation or not')
    parser.add_argument('--Principal_variable_azimuth', type=int, default=20, help='Principal variable azimuth')
    parser.add_argument('--alpha', type=float, default=0.6, help='compress factor')

    parser.add_argument('--save', type=bool, default=False, help='Save the model')
    parser.add_argument('--plot', type=bool, default=True, help='Plot the result')
    args, unknowns = parser.parse_known_args()
    
    if args.device == "auto":
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    return args
