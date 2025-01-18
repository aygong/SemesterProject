import torch
import torch.nn.functional as F
import dgl
import math
import json
import numpy as np
import random

##############################################################################
#                           UTILS (revised)                                  #
##############################################################################

parameters = [
    # index,  [ ID, type, K, N, T, Q, L ]
    ['0', 'a', 2, 16, 480, 3, 30],  # 0
    ['1', 'a', 2, 20, 600, 3, 30],  # 1
    ['2', 'a', 2, 24, 720, 3, 30],  # 2
    ['4', 'a', 3, 24, 480, 3, 30],  # 3
    ['6', 'a', 3, 36, 720, 3, 30],  # 4
    ['9', 'a', 4, 32, 480, 3, 30],  # 5
    ['10', 'a', 4, 40, 600, 3, 30], # 6
    ['11', 'a', 4, 48, 720, 3, 30], # 7
    ['24', 'b', 2, 16, 480, 6, 45], # 8
    ['25', 'b', 2, 20, 600, 6, 45], # 9
    ['26', 'b', 2, 24, 720, 6, 45], # 10
    ['28', 'b', 3, 24, 480, 6, 45], # 11
    ['30', 'b', 3, 36, 720, 6, 45], # 12
    ['33', 'b', 4, 32, 480, 6, 45], # 13
    ['34', 'b', 4, 40, 600, 6, 45], # 14
    ['35', 'b', 4, 48, 720, 6, 45], # 15
]

def node_to_user(N):
    node2user = {}
    for i in range(1, 2 * N + 1):
        if i <= N:
            node2user[i] = i
        else:
            node2user[i] = i - N

    return node2user

def get_device(cuda_available):
    if cuda_available:
        print('CUDA is available. Utilize GPUs for computation.\n')
        device = torch.device("cuda")
    else:
        print('CUDA is not available. Utilize CPUs for computation.\n')
        device = torch.device("cpu")
    return device

def load_instance(index, mode):
    _type_, K, N, T, Q, L = parameters[index][1:]
    if mode == 'train':
        print('Training instances -> Type: {}.'.format(_type_), 'K: {}.'.format(K), 'N: {}.'.format(N),
              'T: {}.'.format(T), 'Q: {}.'.format(Q), 'L: {}.'.format(L), '\n')
    else:
        print('Test instances -> Type: {}.'.format(_type_), 'K: {}.'.format(K), 'N: {}.'.format(N),
              'T: {}.'.format(T), 'Q: {}.'.format(Q), 'L: {}.'.format(L), '\n')

    return _type_, K, N, T, Q, L


def collate(samples):
    """
    Form a mini batch from a given list of samples.
    The input samples is a list of pairs (graph, vehicle_id, action, value).
    """
    graphs, ks, actions, values = map(list, zip(*samples))
    ks = torch.tensor(ks).long()
    actions = torch.tensor(actions).long()
    values = torch.tensor(values)
    batched_graph = dgl.batch(graphs)

    return batched_graph, ks, actions, values

def euclidean_distance(coord_start, coord_end):
    return math.sqrt((coord_start[0] - coord_end[0])**2 + (coord_start[1] - coord_end[1])**2)

def shift_window(time_window, shift):
    """
    Shift a time-window [start, end] by 'shift' units.
    """
    return [
        max(0.0, time_window[0] - shift),
        max(0.0, time_window[1] - shift)
    ]

def check_window(time_window, time):
    return (time < time_window[0]) or (time > time_window[1])

def update_ride_time(vehicle, users, ride_time):
    for uid in vehicle.serving:
        users[uid - 1].ride_time += ride_time

def node_to_user(N):
    """
    Old approach used a dictionary to handle node->user for 2N user-nodes.
    """
    node2user = {}
    for i in range(1, 2*N + 1):
        if i <= N:
            node2user[i] = i
        else:
            node2user[i] = i - N
    return node2user

def one_hot_node_type(ntype):
    """
    Map node types to index in a one-hot vector
    We'll store them in the first dimension of node_features
    """
    if ntype=='pickup':      return 0
    elif ntype=='dropoff':   return 1
    elif ntype=='wait':      return 2
    elif ntype=='source':    return 3
    elif ntype=='destination': return 4
    else:
        raise ValueError(f"Unknown node type {ntype}")

########################################################################
#  The is_edge logic for building the state graph. We re-run this each #
#  step to update feasibility bits in the edge data.                   #
########################################################################

def active_vehicles(darp):
    return sum([v.free_time < 1440 for v in darp.vehicles])

def waiting_users(darp):
    return sum([u.alpha == 0 for u in darp.users])

def is_edge(darp, i_u, u, k_u, t_u, u_next, i_v, v, k_v, t_v, v_next):
    """
    Return True if an edge i_u->i_v should be active in the state graph.
    (Same logic as your old code, but we keep the node indexing consistent.)
    """
    # 1) If same node
    if i_u == i_v:
        return False
    
    # 2) If not feasible in terms of arc elimination (time windows, capacity, etc.)
    #    We require that at least i_u->i_v or i_v->i_u is feasible:
    if not (darp.is_arc_feasible(i_u, i_v) or darp.is_arc_feasible(i_v, i_u)):
        return False
    
    # 3) If a user has alpha=2 => fully served => typically no edges unless a vehicle is there
    if (u and u.alpha == 2 and not k_u) or (v and v.alpha == 2 and not k_v):
        return False
    
    # 4) If pickup is already visited => alpha=1 or alpha=2 => skip unless a vehicle is here
    if (t_u=='pickup' and u.alpha==1 and not k_u) or (t_v=='pickup' and v.alpha==1 and not k_v):
        return False
    
    # 5) If it's an empty source station => no edges
    if (t_u=='source' and not k_u) or (t_v=='source' and not k_v):
        return False
    
    # 6) The waiting node logic
    #    If either node is 'wait', we connect to everything except certain constraints:
    if t_u=='wait' or t_v=='wait':
        # If the 'next vehicle node' is a dropoff, maybe skip
        if (u_next and t_u=='dropoff') or (v_next and t_v=='dropoff'):
            return False
        # If the 'next vehicle node' is a pickup and waiting would break the window, skip
        if (u_next and t_u=='pickup' and k_u.free_time + darp.args.wait_time > u.pickup_window[1]):
            return False
        if (v_next and t_v=='pickup' and k_v.free_time + darp.args.wait_time > v.pickup_window[1]):
            return False
        return True
    
    # 7) Destination logic
    if t_u=='destination':
        if k_v:
            # If last active vehicle but not all users served => can't go to destination
            if (active_vehicles(darp) == 1 and waiting_users(darp) != 0):
                return False
            # If vehicle is serving last user => might be allowed
            if t_v=='dropoff' and len(k_v.serving) <= 1:
                return True
            if t_v=='source': # connecting destination->source is sometimes allowed
                return True
            return False
        if t_v=='dropoff': # connect destination to dropoff
            return True
        return False
    
    # If t_u=='source' and vehicle present
    if t_u=='source' and k_u:
        if t_v=='pickup' and (not k_v) and (k_u.free_capacity >= v.load):
            return True
        if t_v=='destination':
            # if last active vehicle => can't go to destination if users remain
            if (active_vehicles(darp)==1 and waiting_users(darp)!=0):
                return False
            return True
        return False
    
    # If t_u=='pickup'
    if t_u=='pickup':
        if k_u:
            # pickup->pickup feasible if capacity
            if t_v=='pickup' and (not k_v) and (k_u.free_capacity >= v.load):
                return True
            # pickup->dropoff feasible if dropoff belongs to same user or is in 'serving'
            if t_v=='dropoff' and not k_v:
                return (v.id in k_u.serving or u==v)
            return False
        else:
            # if there's a vehicle at v:
            if k_v:
                # connect pickup->(source/pickup/dropoff) if capacity
                if t_v in ['source','pickup','dropoff'] and (k_v.free_capacity >= u.load):
                    return True
                return False
            else:
                # no vehicles => possibly connect to everything except destination
                return (t_v != 'destination')
    
    # If t_u=='dropoff'
    if t_u=='dropoff':
        if k_u:
            # dropoff->pickup if capacity
            if t_v=='pickup' and (k_u.free_capacity >= v.load):
                return True
            # dropoff->dropoff if user belongs to that vehicle
            if t_v=='dropoff':
                return (v.id in k_u.serving)
            # dropoff->destination if done serving
            if (t_v=='destination') and (len(k_u.serving) <= 1):
                # if last active vehicle => can't go destination if users remain
                if (active_vehicles(darp)==1 and waiting_users(darp)!=0):
                    return False
                return True
            return False
        else:
            if k_v:
                # dropoff->(some node) if it is serving that user or it's the same user
                return (u.id in k_v.serving or (t_v=='pickup' and u==v))
            else:
                # no vehicles => connect to everything except source?
                return (t_v != 'source')
    
    raise RuntimeError("End of is_edge reached without return")
