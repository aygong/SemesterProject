import math
import torch
import dgl
import random

parameters = [['0', 'a', 2, 16, 480, 3, 30],  # 0
              ['1', 'a', 2, 20, 600, 3, 30],  # 1
              ['2', 'a', 2, 24, 720, 3, 30],  # 2
              ['4', 'a', 3, 24, 480, 3, 30],  # 3
              ['6', 'a', 3, 36, 720, 3, 30],  # 4
              ['9', 'a', 4, 32, 480, 3, 30],  # 5
              ['10', 'a', 4, 40, 600, 3, 30],  # 6
              ['11', 'a', 4, 48, 720, 3, 30],  # 7
              ['24', 'b', 2, 16, 480, 6, 45],  # 8
              ['25', 'b', 2, 20, 600, 6, 45],  # 9
              ['26', 'b', 2, 24, 720, 6, 45],  # 10
              ['28', 'b', 3, 24, 480, 6, 45],  # 11
              ['30', 'b', 3, 36, 720, 6, 45],  # 12
              ['33', 'b', 4, 32, 480, 6, 45],  # 13
              ['34', 'b', 4, 40, 600, 6, 45],  # 14
              ['35', 'b', 4, 48, 720, 6, 45]]  # 15


def euclidean_distance(coord_start, coord_end):
    return math.sqrt((coord_start[0] - coord_end[0]) ** 2 + (coord_start[1] - coord_end[1]) ** 2)


def shift_window(time_window, time):
    return [max(0.0, time_window[0] - time), max(0.0, time_window[1] - time)]


def check_window(time_window, time):
    return time < time_window[0] or time > time_window[1]


def update_ride_time(vehicle, users, ride_time):
    for uid in vehicle.serving:
        users[uid - 1].ride_time += ride_time


def load_instance(index, mode):
    _type_, K, N, T, Q, L = parameters[index][1:]
    if mode == 'train':
        print('Training instances -> Type: {}.'.format(_type_), 'K: {}.'.format(K), 'N: {}.'.format(N),
              'T: {}.'.format(T), 'Q: {}.'.format(Q), 'L: {}.'.format(L), '\n')
    else:
        print('Test instances -> Type: {}.'.format(_type_), 'K: {}.'.format(K), 'N: {}.'.format(N),
              'T: {}.'.format(T), 'Q: {}.'.format(Q), 'L: {}.'.format(L), '\n')

    return _type_, K, N, T, Q, L


def get_device(cuda_available):
    if cuda_available:
        print('CUDA is available. Utilize GPUs for computation.\n')
        device = torch.device("cuda")
    else:
        print('CUDA is not available. Utilize CPUs for computation.\n')
        device = torch.device("cpu")
    return device


def node_to_user(N):
    node2user = {}
    for i in range(1, 2 * N + 1):
        if i <= N:
            node2user[i] = i
        else:
            node2user[i] = i - N

    return node2user

def one_hot_node_type(typ):
    if typ=='pickup':
        return 0
    elif typ=='dropoff':
        return 1
    elif typ=='wait':
        return 2
    elif typ=='source':
        return 3
    elif typ=='destination':
        return 4
    else:
        raise ValueError(f'Unknown node type: {typ}.')
    
def active_vehicles(darp):
    """
    returns the number of active vehicles
    """
    return sum([v.free_time < 1440 for v in darp.vehicles])

def waiting_users(darp):
    """
    returns the number of waiting users (alpha=0)
    """
    return sum([u.alpha==0 for u in darp.users])
    
def is_edge(darp, i_u, u, k_u, t_u, u_next, i_v, v, k_v, t_v, v_next):
    """
    For state graph computation.
    Returns True if an edge should be drawn between the two node, False otherwise.
    """
    if i_u == i_v:
        return False
    if not (darp.is_arc_feasible(i_u, i_v) or darp.is_arc_feasible(i_v, i_u)):
        return False
    if (u and u.alpha == 2 and not k_u) or (v and v.alpha == 2 and not k_v): # already visited user
        return False
    if (t_u == 'pickup' and u.alpha == 1 and not k_u) or (t_v == 'pickup' and v.alpha == 1 and not k_v): # already visited pickups
        return False
    if (t_u == 'source' and not k_u) or (t_v == 'source' and not k_v): # empty source station
        return False
    if t_u == 'wait' or t_v == 'wait': # waiting node connected to every other node, CHANGE ???
        if (u_next and t_u == 'dropoff') or (v_next and t_v == 'dropoff'):
            # do not connect to wait if next vehicle is on dropoff
            return False
        if (u_next and t_u == 'pickup' and k_u.free_time + darp.args.wait_time > u.pickup_window[1]) or (v_next and t_v == 'pickup' and k_v.free_time + darp.args.wait_time > v.pickup_window[1]):
            # do not connect to wait if user window is going to be broken
            return False
        return True
    if k_u and k_v: # both contain vehicles
        return False
    

    
    if t_u == 'destination':
        if k_v:
            if active_vehicles(darp) == 1 and waiting_users(darp) != 0:
                # if k_v is the last active vehicle and there are still users to be picked up, it cannot go to destination
                return False
            if t_v == 'dropoff' and len(k_v.serving) <= 1: # vehicle can serve last user and leave
                return True
            if t_v == 'source': # Vehicle at source is empty
                return True
            return False
        if t_v == 'dropoff': # connect destination to dropoffs
            return True
        return False
    
    if t_u == 'source' and k_u:
        if t_v == 'pickup' and not k_v and k_u.free_capacity >= v.load: # connect to available pickups
            return True
        if t_v == 'destination':
            if active_vehicles(darp) == 1 and waiting_users(darp) != 0:
                # if k_u is the last active vehicle and there are still users to be picked up, it cannot go to destination
                return False
            return True
        return False
    
    if t_u == 'pickup':
        if k_u:
            if t_v == 'pickup' and not k_v and k_u.free_capacity >= v.load: # connect to available pickups
                return True
            if t_v == 'dropoff' and not k_v: # connect to available dropoffs
                return (v.id in k_u.serving or u==v)
            return False
        else:
            if k_v:
                if (t_v=='source' or t_v=='pickup' or t_v=='dropoff') and k_v.free_capacity >= u.load: # connect to source if vehicle there
                    return True
                return False
            else:
                return t_v != 'destination'
    
    if t_u == 'dropoff':
        if k_u:
            if t_v=='pickup' and k_u.free_capacity >= v.load:
                return True
            if t_v == 'dropoff':
                return (v.id in k_u.serving)
            if t_v == 'destination' and len(k_u.serving) <= 1:
                if active_vehicles(darp) == 1 and waiting_users(darp) != 0:
                    # if k_u is the last active vehicle and there are still users to be picked up, it cannot go to destination
                    return False
                return True
            return False
        else:
            if k_v:
                return (u.id in k_v.serving or (t_v=='pickup' and u==v))
            else:
                return t_v != 'source'
            
    raise RuntimeError('End of is_edge function without returning') # Should not happen but sanity check

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

def shuffle_list(*ls):
    """ Shuffles all the given lists with the same random permuation"""
    l =list(zip(*ls))

    random.shuffle(l)
    return zip(*l)