import argparse
# vessl_on
# map_name1 = '6h_vs_8z'
# GNN = 'GAT'
def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vessl", type=bool, default=False, help="vessl AI 사용여부")
    parser.add_argument("--map_name", type=str, default='6h_vs_8z', help="map name")
    parser.add_argument("--GNN", type=str, default='GAT', help="map name")
    parser.add_argument("--hidden_size_obs", type=int, default=32, help="GTN 해당")
    parser.add_argument("--hidden_size_comm", type=int, default=56, help="")
    parser.add_argument("--n_representation_job", type=int, default=15, help="GTN 해당")
    parser.add_argument("--n_representation_machine", type=int, default=32, help="")
    parser.add_argument("--hidden_size_Q", type=int, default=128, help="GTN 해당")
    parser.add_argument("--hidden_size_meta_path", type=int, default=42, help="GTN 해당")
    parser.add_argument("--buffer_size", type=int, default=50000, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--teleport_probability", type=float, default=0.9, help="teleport_probability")
    parser.add_argument("--gtn_beta", type=float, default=0.05, help="teleport_probability")
    parser.add_argument("--gamma", type=float, default=0.8, help="discount ratio")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--n_multi_head", type=int, default=1, help="number of multi head")
    parser.add_argument("--dropout", type=float, default=0.6, help="dropout")
    parser.add_argument("--num_episode", type=int, default=1000000, help="number of episode")
    parser.add_argument("--train_start", type=int, default=1, help="number of train start")
    parser.add_argument("--epsilon", type=float, default=0.5, help="initial value of epsilon greedy")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="minimum value of epsilon greedy")
    parser.add_argument("--anneal_steps", type=int, default=100000, help="anneal ratio of epsilon greedy")
    return parser.parse_args()