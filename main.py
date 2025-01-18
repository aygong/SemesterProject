from dataset import *
from supervision import *
from evaluation import *
from reinforcement import * 
from reinforcement_PPO import reinforce_PPO
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Arguments of dataset.py
    parser.add_argument('--dataset', type=int, default=0) # 0 for false, 1 for true
    parser.add_argument('--train_index', type=int, default=0)
    parser.add_argument('--num_sl_subsets', type=int, default=100) 
    parser.add_argument('--num_sl_instances', type=int, default=500)
    parser.add_argument('--wait_time', type=int, default=7)
    parser.add_argument('--arc_elimination', type=int, default=0)

    # Arguments of supervision.py
    parser.add_argument('--supervision', type=int, default=0) # 0 for false, 1 for true
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--loss_ratio', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)


    # Arguments of reinforcement.py
    parser.add_argument('--reinforcement', type=int, default=0) # 0 for false, 1 for true
    parser.add_argument('--num_rl_instances', type=int, default=5)

    # Arguments of evaluation.py
    parser.add_argument('--evaluation', type=int, default=1) # 0 for false, 1 for true
    parser.add_argument('--model_type', type=int, default=0)  # 1: RL, 0: SL
    parser.add_argument('--test_index', type=int, default=1)
    parser.add_argument('--num_tt_instances', type=int, default=100)
    parser.add_argument('--beam', type=int, default=0)

    # Arguments of transformer.py
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pe_dim', type=int, default=10) # laplacian positional encoding, 0 for no positional encoding

    # Argument for saving
    parser.add_argument('--filename_index', type=int, default=3)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    if args.dataset:
        print("#################################################")
        print("########## Dataset generation started. ##########")
        print("#################################################\n")
        dataset(args)

    if args.supervision:
        print("##################################################")
        print("########## Supervised learning started. ##########")
        print("##################################################\n")
        supervision(args)

    if args.reinforcement: 
        print("#####################################################")
        print("########## Reinforcement learning started. ##########")
        print("#####################################################\n")
        reinforce_PPO(args)

    if args.evaluation:
        print("#########################################")
        print("########## Evaluation started. ##########")
        print("#########################################\n")
        evaluation(args)

        

if __name__ == '__main__':
    main()
