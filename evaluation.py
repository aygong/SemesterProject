from environment import *


import os
import time as t
from collections import deque
import copy
import torch


from graph_transformer import GraphTransformerNet

def evaluation(args, model=None):
    # Determine if your system supports CUDA

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    darp = Darp(args, mode='evaluate', device=device)
    num_nodes = 2*darp.train_N + darp.train_K + 2
    num_edge_feat = 5 if args.arc_elimination else 3 # include feasibility as feature when doing arc elimination

    if model == None:

        darp.model = GraphTransformerNet(
            device=device,
            num_nodes=num_nodes,
            num_node_feat=17,
            num_edge_feat=num_edge_feat,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_k=args.d_k,
            d_v=args.d_v,
            d_ff=args.d_ff,
            dropout=0.1
        )

        # Load the trained model
        model_name = darp.train_name + '-' + str(args.wait_time) +'-'+ str(args.filename_index)

        if args.model_type:
            model = "rl"
            print("Load the model trained by reinforcement learning.\n")
        else:
            model = "sl"
            print("Load the model trained by supervised learning.\n")
        # model = "ppo"
        #name = './model/ppo-a2-16-7-3-'+str(i)+'.model'

        checkpoint = torch.load('./model/' + model + '-' + model_name + '.model', map_location=device)
        #checkpoint = torch.load(name, map_location=device)

        #print('Load the model from' , name)
        darp.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        darp.model=model

    darp.model.eval()


    darp.model.cuda()

    path_result = './result/'

    # Initialize the lists of metrics
    eval_run_time = []
    eval_time_penalty = []
    eval_rist_cost = []
    eval_pred_cost = []
    eval_window = []
    eval_ride_time = []
    eval_not_same = []
    eval_not_done = []
    eval_rela_diff = []

    # Set 'src_mask'
    src_mask = [1 for _ in range(darp.test_N)] + [0 for _ in range(darp.train_N - darp.test_N)]
    src_mask = torch.Tensor(src_mask).to(device)

    for num_instance in range(args.num_tt_instances):
        print('--------Evaluation on Instance {}:--------'.format(num_instance + 1))
        start = t.time()
        true_cost = darp.reset(num_instance)

        if args.beam:
            darps = beam_search(darp, num_instance, src_mask, args.beam)
            print('\n--------Beam results:--------')
            for darp in darps:
                print(round(darp[0].cost(), 2), round(abs(true_cost - darp[0].cost()) / true_cost * 100, 2),
                      len(darp[0].break_window), len(set(darp[0].break_ride_time)), round(darp[0].time_penalty, 2))
            print()
            darp, darp_cost = beam_choose(darps)
        else:
            darp_cost = greedy_evaluation(darp, num_instance, src_mask)
        end = t.time()

        # Evaluate the predicted results
        for k in range(0, darp.test_K):
            vehicle = darp.vehicles[k]
            print('> Vehicle {}'.format(vehicle.id))
            # Apply the node-user mapping
            for r, node in enumerate(vehicle.route):
                if 0 < node < 2 * darp.test_N + 1:
                    vehicle.route[r] = darp.node2user[node]
            # Print the Rist's route of the vehicle
            rist_route = zip(vehicle.route[1:-1], vehicle.schedule[1:-1])
            print('Rist\'s route:', [term[0] for term in rist_route])
            # Print the predicted route of the vehicle
            pred_route = zip(vehicle.pred_route[1:-1], vehicle.pred_schedule[1:-1])
            print('Predicted route:', [term[0] for term in pred_route])

        run_time = end - start
        # Append the lists of metrics
        eval_rist_cost.append(true_cost)
        eval_pred_cost.append(darp_cost)
        eval_window.append(len(darp.break_window))
        eval_ride_time.append(len(set(darp.break_ride_time)))
        eval_not_same.append(len(darp.break_same))
        eval_not_done.append(len(darp.break_done))
        eval_rela_diff.append((darp_cost - true_cost) / true_cost * 100)
        eval_run_time.append(run_time)
        eval_time_penalty.append(darp.time_penalty)

        # Print the Rist's cost, the predicted cost, and the relative difference
        print('> Objective')
        print('Rist\'s cost: {:.4f}'.format(eval_rist_cost[-1]))
        print('Predicted cost: {:.4f}'.format(eval_pred_cost[-1]))
        print('Relative difference (%): {:.2f}'.format(eval_rela_diff[-1]))

        # Print the number of broken constraints
        print('> Constraint')
        print('# broken time window: {}'.format(eval_window[-1]))
        print('# broken ride time: {}\n'.format(eval_ride_time[-1]))

        with open(path_result + 'evaluation_log.txt', 'a+') as file:
            json.dump({
                'Num instance': num_instance,
                'Rist\'s cost': '{:.4f}'.format(eval_rist_cost[-1]),
                'Predicted cost': '{:.4f}'.format(eval_pred_cost[-1]),
                '# broken time window': eval_window[-1],
                '# broken ride time': eval_ride_time[-1],
                'run time': eval_run_time[-1],
            }, file)
            file.write("\n")

    os.makedirs(path_result, exist_ok=True)

    standard_dict = {
            'Cost (Rist 2021)': round(eval_rist_cost[0], 2),
            'Cost (predicted)': round(eval_pred_cost[0], 2),
            'Diff. (%)': round(eval_rela_diff[0], 2),
            '# Time Window': eval_window[0],
            '# Ride Time': eval_ride_time[0],
            'Time penalty': round(eval_time_penalty[0], 2),
            'Run time': round(eval_run_time[0], 2),
        }
    
    average_dict = {
            'Aver. Cost (Rist 2021)': round(sum(eval_rist_cost) / len(eval_rist_cost), 2),
            'Aver. Cost (predicted)': round(sum(eval_pred_cost) / len(eval_pred_cost), 2),
            'Aver. Diff. (%)': round(sum(eval_rela_diff) / len(eval_rela_diff), 2),
            'Aver. # Time Window': round(sum(eval_window) / len(eval_window), 2),
            'Aver. # Ride Time': round(sum(eval_ride_time) / len(eval_ride_time), 2),
            'Aver. Time penalty': round(sum(eval_time_penalty) / len(eval_time_penalty), 2),
            'Aver. Run time': round(sum(eval_run_time) / len(eval_run_time), 2),
        }
    
    not_same = {
            '# Not Same': int(np.sum(np.asarray(eval_not_same) > 0)),
            '# Not Done': int(np.sum(np.asarray(eval_not_done) > 0)),
        }
    
     # Print the metrics on one standard instance
    print('--------Metrics on one standard instance:--------')
    for k,v in standard_dict.items():
        print(k,':',v)

    # Print the metrics on multiple random instances
    print('--------Average metrics on {} random instances:--------'.format(args.num_tt_instances))
    for k,v in average_dict.items():
        print(k,':',v)

    # Print the number of problematic requests
    for k,v in not_same.items():
        print(k,':',v)

    with open(path_result + 'evaluation.txt', 'a+') as output:
        # Dump the parameters of training instances
        output.write('Training instances -> ')
        json.dump({
            'Type': darp.train_type, 'K': darp.train_K, 'N': darp.train_N,
            'T': darp.train_T, 'Q': darp.train_Q, 'L': darp.train_L,
        }, output)
        output.write('\n')

        # Dump the parameters of test instances
        output.write('Test instances -> ')
        json.dump({
            'Type': darp.test_type, 'K': darp.test_K, 'N': darp.test_N,
            'T': darp.test_T, 'Q': darp.test_Q, 'L': darp.test_L,
        }, output)
        output.write('\n')

        # Dump the metrics on one standard instance
        json.dump(standard_dict, output)
        output.write('\n')

        # Dump the metrics on multiple random instances
        json.dump(average_dict, output)
        output.write('\n')

        # Dump the number of problematic requests
        json.dump(not_same, output)
        output.write('\n')
    

    with open(f'{path_result}{model_name}-{darp.test_K}-{darp.test_N}-evaluation.txt', 'w') as output:
        
        # Dump the metrics on one standard instance
        json.dump(standard_dict, output)
        output.write('\n')

        # Dump the metrics on multiple random instances
        json.dump(average_dict, output)
        output.write('\n')

        # Dump the number of problematic requests
        json.dump(not_same, output)
        output.write('\n')

    return average_dict


def greedy_evaluation(darp, num_instance, src_mask=None, logs=True):
    # Run the simulator
    darp.log_probs = []
    while darp.finish():
        # Select next available vehicle
        free_times = [vehicle.free_time for vehicle in darp.vehicles]
        time = np.min(free_times)
        indices = np.argwhere(free_times == time)
        indices = indices.flatten().tolist()
        for _, k in enumerate(indices):
            if darp.vehicles[k].free_time >= 1440:
                continue
            darp.beta(k) # Update beta
            state, next_vehicle_node = darp.state_graph(k, time) # Compute state
            action_node, probs = darp.predict(state, next_vehicle_node, user_mask=None, src_mask=src_mask) # Predict action
            action = darp.node2action(action_node) # Corresponding node
            darp.log_probs.append(torch.log(probs.squeeze(0)[action]))
            darp.evaluate_step(k, action) # Simulate one step of MDP



    return darp.cost()

def add_candidates(envs, env, state, k_best_new, beam_width, probs, k, next_vehicle_node, idx, score, n_broken):
    """
    Add candidate actions with best probabilities, removing the waiting action. Used in beam search.
    envs: dict from indices to corresponding darp
    env: current darp
    state: current state graph
    k_best_new: list that keeps track of all the new candidates
    probs: probabilities of each action
    k: indice of next vehicle
    next_vehicle_node: node containing next vehicle
    idx: indice of current darp
    score: probability score of the current path
    n_broken: current number of broken constraints
    """
    log_probs, action_nodes = torch.topk(torch.log(probs.squeeze(0)[1:]), min(beam_width, len(state.successors(next_vehicle_node))-1)) # 1: to not take into account waiting node

    envs[idx] = copy.deepcopy(env)

    for log_prob, action_node in zip(log_probs, action_nodes):
        action_node += 1 # Shift because we removed waiting action
        # expand each current candidate
        action = env.node2action(action_node)
        k_best_new.append((idx, score - log_prob.item(), k, action.item(), n_broken))
    

def beam_search(darp, num_instance, src_mask, beam_width):
    """
    Beam search algorithm for the DARP problem. Maintain 
    the best beam_width solutions at each time step and expand them to the next time step.
    WARNING: Increases running time exponentially with beam_width.
    a2-16 greedy takes ~10s while beam search with beam_width=10 takes ~1000s.
    """
    # TODO: transpositions are possible, they need to be detected and removed from the beam
    k_best = [(darp, False, 0.0, 0)]  # (darp, finish, sumlogprob, n_broken_constraints)
    # Run the simulator
    while sum([done for (env, done, score, n_broken) in k_best]) < beam_width:
        k_best_new = []
        envs = {}
        i = 0
        for env, done, score, n_broken in k_best:
            if not done:
                waiting = True
                waited_too_much = False
                wait_count_per_vehicle = np.zeros(len(env.vehicles))
                while waiting:

                    free_times = [vehicle.free_time for vehicle in env.vehicles]
                    time = np.min(free_times)
                    indices = np.argwhere(free_times == time)
                    indices = indices.flatten().tolist()

                    k = indices[0]
                    if env.vehicles[k].free_time == 1440:
                        if sum(wait_count_per_vehicle) > 0:
                            waited_too_much = True
                            break
                        else:
                            raise RuntimeError(f'Environment should be done if next free time is 1440, free_times: {free_times}, wait_count: {wait_count_per_vehicle}')

                    env.beta(k)
                    state, next_vehicle_node = env.state_graph(k, time)
                    action_node, probs = env.predict(state, next_vehicle_node, user_mask=None, src_mask=src_mask)
                    action = env.node2action(action_node)

                    if action == env.train_N + 1: # Waiting action
                        if wait_count_per_vehicle[k] == 0:
                            # Save other actions to keep possibility of not waiting
                            add_candidates(envs, env, state, k_best_new, beam_width, probs, k, next_vehicle_node, i, score, n_broken)
                            i += 1
                        env.evaluate_step(k, action)
                        wait_count_per_vehicle[k] += 1
                    else:
                        waiting = False
                if not waited_too_much:
                    add_candidates(envs, env, state, k_best_new, beam_width, probs, k, next_vehicle_node, i, score, n_broken)
                    i += 1

        # order by score, select k best
        k_best_new = sorted(k_best_new, key=lambda x: (x[4], x[1]))[:beam_width]

        # step the env in potential envs
        k_best = []
        for (i, score, k, action, _) in k_best_new:
            env = copy.deepcopy(envs[i])
            env.evaluate_step(k, action)
            n_broken_constraints = len(env.break_window) + len(env.break_ride_time) + 2*len(env.break_same) + 2*len(env.break_done)
            k_best.append((env, not env.finish(), score, n_broken_constraints))
    return k_best


def beam_choose(darps):
    best_darp = sorted(darps, key=lambda x: (x[3], x[0].cost()))[0]
    return best_darp[0], best_darp[0].cost()


