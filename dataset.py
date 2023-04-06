from environment import *

import os
import shutil
import sys


def dataset(args):
    darp = Darp(args, mode='supervise')

    path_dataset = './dataset/'
    os.makedirs(path_dataset, exist_ok=True)
    # shutil.rmtree(path_dataset)
    # print("Directory {} has been removed successfully".format(path_dataset))
    # os.makedirs(path_dataset)

    data = []
    num_dataset = 1

    for num_instance in range(len(darp.list_instances)):
        objective = darp.reset(num_instance)

        sum_travel_times = 0
        # Run the simulator
        while darp.finish():
            free_times = [vehicle.free_time for vehicle in darp.vehicles]
            time = np.min(free_times)
            indices = np.argwhere(free_times == time)
            indices = indices.flatten().tolist()

            for _, k in enumerate(indices):
                if darp.vehicles[k].free_time == 1440:
                    continue

                darp.beta(k)
                #state = darp.state(k, time)
                state, next_vehicle_node = darp.state_graph(k, time)
                
                
                cost_to_go = objective - sum_travel_times # cost to go from this state until the end, BEFORE taking the action
                action = darp.action(k)
                node = darp.action2node(action)
                if node not in state.successors(next_vehicle_node):
                    #print((state.edges()[0]).tolist())
                    #print((state.edges()[1]).tolist())
                    #print(state.successors(next_vehicle_node))
                    #print(state.ndata['feat'][next_vehicle_node])
                    #print(state.ndata['feat'][darp.action2node(action)])
                    #print(next_vehicle_node)
                    #print(action)
                    #print(darp.action2node(action))
                    #print(darp.vehicles[k].free_capacity)
                    #print(darp.users[action].load, darp.users[action].alpha)
                    #print(darp.vehicles[k].serving)
                    raise ValueError('Error in graph creation: vehicle cannot perform best action.')
                #print(action)
                
                #print(f'vehicle id: {k}, vehicle node: {next_vehicle_node}')
                #print('vehicle state: ', state.ndata['feat'][next_vehicle_node])
                #print('vehicle neighbors: ', state.successors(next_vehicle_node))
                
                #print(f'time: {time}')
                #print(f'action: {action}, action_node: {node}')
                #print('users alphas: ', [u.alpha for u in darp.users])
                #print('users ids: ', [u.id for u in darp.users])
                #print(len(state.successors(0)))
                #print('-'*30)
                
                for suc in state.successors(next_vehicle_node):
                    if suc > darp.train_N and suc <= darp.train_N * 2:
                        u_id = suc - (darp.train_N + 1)
                        u = darp.users[u_id]
                        v = darp.vehicles[k]
                        if u.served != v.id:
                            raise ValueError(f'Error in graph creation: vehicle is not supposed to be linked to other dropoffs. vehicle node: {next_vehicle_node}, user node: {suc}')
                    
                    if next_vehicle_node > 0 and next_vehicle_node <= darp.train_N and suc == 2*darp.train_N + 1:
                        raise ValueError(f'Error in graph creation: vehicle on pickup is not supposed to be linked to destination. vehicle node: {next_vehicle_node}, user node: {suc}')
                    
                if next_vehicle_node > 0 and next_vehicle_node <= darp.train_N:
                    dropoff_node = next_vehicle_node + darp.train_N
                    if dropoff_node not in state.successors(next_vehicle_node):
                        raise ValueError(f'Error in graph creation: vehicle on pickup is supposed to be linked to corresponding dropoff. vehicle node: {next_vehicle_node}, user node: {dropoff_node}')


                sum_travel_times += darp.supervise_step(k)


                data.append([state, next_vehicle_node, node, cost_to_go])

        # Save the training sets
        print(num_dataset, num_instance + 1, sys.getsizeof(data), len(data), objective)
        if (num_instance + 1) % args.num_sl_instances == 0:
            file = 'dataset-' + darp.train_name + '-' + str(num_dataset) + '.pt'
            print('Save {}.\n'.format(file))
            torch.save(data, path_dataset + file)
            data = []
            num_dataset += 1
            if num_dataset > args.num_sl_subsets:
                break
