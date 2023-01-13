from dataset import *
from supervision import *
from evaluation import *
from reinforcement import *
<<<<<<< HEAD

import argparse
=======
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Arguments of dataset.py
    parser.add_argument('--dataset', type=bool, default=True)
    parser.add_argument('--train_index', type=int, default=9)
    parser.add_argument('--num_sl_subsets', type=int, default=1)
    parser.add_argument('--num_sl_instances', type=int, default=500)
    parser.add_argument('--wait_time', type=int, default=7)

    # Arguments of supervision.py
    parser.add_argument('--supervision', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)

    # Arguments of reinforcement.py
    parser.add_argument('--reinforcement', type=bool, default=True)
    parser.add_argument('--num_rl_instances', type=int, default=5)

    # Arguments of evaluation.py
    parser.add_argument('--evaluation', type=bool, default=True)
    parser.add_argument('--model_type', type=bool, default=True)  # True: RL, False: SL
    parser.add_argument('--test_index', type=int, default=8)
<<<<<<< HEAD
    parser.add_argument('--num_tt_instances', type=int, default=100)
    parser.add_argument('--beam', type=int, default=0)
=======
    parser.add_argument('--num_test_instances', type=int, default=10)
    parser.add_argument('--beam', type=int, default=0)
    parser.add_argument('--rl_flag', type=bool, default=False)
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871

    # Arguments of transformer.py
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Arguments of reinforcement.py
    parser.add_argument('--reinforcement', type=bool, default=False)
    parser.add_argument('--num_rl_instances', type=int, default=100)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
<<<<<<< HEAD

    if bool(args.dataset):
        print("#################################################")
        print("########## Dataset generation started. ##########")
        print("#################################################\n")
        dataset(args)

    if bool(args.supervision):
        print("##################################################")
        print("########## Supervised learning started. ##########")
        print("##################################################\n")
        supervision(args)

    if bool(args.reinforcement):
        print("#####################################################")
        print("########## Reinforcement learning started. ##########")
        print("#####################################################\n")
        reinforce(args)

    if bool(args.evaluation):
        print("#########################################")
        print("########## Evaluation started. ##########")
        print("#########################################\n")
=======
    print(args.dataset)
    if bool(args.dataset):
        print("Dataset generation started.")
        dataset(args)

    if bool(args.supervision):
        print("Supervision started.")
        supervision(args)

    if bool(args.reinforcement):
        print("Reinforcement learning started.")
        reinforce(args)

    if bool(args.evaluation):
        print("Evaluation started.")
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871
        evaluation(args)
    


if __name__ == '__main__':
    main()
