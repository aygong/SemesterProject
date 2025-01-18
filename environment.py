from utils import *

import numpy as np
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as f

import dgl
class Node:
    def __init__(self):
        self.coords = [0.0, 0.0]
        self.serve_duration = 0
        self.load = 0
        self.window = [0, 0]

class User:
    def __init__(self, mode):
        self.id = 0
        self.pickup_coords = [0.0, 0.0]
        self.dropoff_coords = [0.0, 0.0]
        self.serve_duration = 0
        self.load = 0
        self.pickup_window = [0, 0]
        self.dropoff_window = [0, 0]
        self.ride_time = 0.0
        # alpha: 0->waiting, 1->in vehicle, 2->finished
        self.alpha = 0
        # beta:  0->waiting, 1->can be served by current vehicle, 2->cannot be served
        self.beta = 0
        # Which vehicle is serving
        self.served = 0
        if mode != 'supervise':
            self.pred_served = []

class Vehicle:
    def __init__(self, mode):
        self.id = 0
        self.route = []
        self.schedule = []
        self.ordinal = 0
        self.coords = [0.0, 0.0]
        self.serving = []
        self.free_capacity = 0
        self.free_time = 0.0
        self.serve_duration = 0
        if mode != 'supervise':
            self.pred_route = [0]
            self.pred_schedule = [0]
            self.pred_cost = 0.0


class Darp:
    def __init__(self, args, mode, device=None):
        super(Darp, self).__init__()

        self.args = args
        self.mode = mode
        self.device = device
        self.model = None
        self.logs = True
        self.log_probs = None

        # Load the parameters of training instances
        self.train_type, self.train_K, self.train_N, self.train_T, self.train_Q, self.train_L = \
            load_instance(args.train_index, 'train')
        # Set the name of training instances
        self.train_name = self.train_type + str(self.train_K) + '-' + str(self.train_N)

        if self.mode != 'evaluate':
            # Get the node-user mapping dictionary of training instances
            self.node2user = node_to_user(self.train_N)
            # Set the path of training instances
            self.data_path = './instance/' + self.train_name + '-train' + '.txt'
        else:
            # Load the parameters of test instances
            self.test_type, self.test_K, self.test_N, self.test_T, self.test_Q, self.test_L = \
                load_instance(args.test_index, 'test')
            
            # Update train N and K if we test on bigger instances, to make it look like it was trained on such instances
            N = max(self.train_N, self.test_N)
            K = max(self.train_K, self.test_K)
            self.train_N = N
            self.train_K = K
            # Get the node-user mapping dictionary of test instances
            self.node2user = node_to_user(N) 
            # Set the name of test instances
            self.test_name = self.test_type + str(self.test_K) + '-' + str(self.test_N)
            # Set the path of test instances
            self.data_path = './instance/' + self.test_name + '-test' + '.txt'

        # Load instances
        self.list_instances = []
        self.load_from_file()

        # Initialize the lists of vehicles and users
        self.users = []
        self.vehicles = []

        if self.mode != 'supervise':
            # Initialize the lists of metrics
            self.break_window = []
            self.break_ride_time = []
            self.break_same = []
            self.break_done = []
            self.time_penalty = 0
            self.indices = []  # for beam search
            self.time = 0.0

        self.graph_initialized = False
        self.num_nodes = 2*self.train_N + self.train_K + 2

    def load_from_file(self, num_instance=None):
        """ Load the instances from the file, in beam search we load the instances one by one """
        if num_instance:
            instance = self.list_instances[num_instance]
            self.list_instances = [instance]
        else:
            with open(self.data_path, 'r') as file:
                for instance in file:
                    self.list_instances.append(json.loads(instance))


    def reset(self, num_instance):
        K, N, T, Q, L = self.parameter()
        instance = self.list_instances[num_instance]

        # 1) Re-initialize users
        self.users = []
        for i in range(1, N + 1):
            user = User(mode=self.mode)
            user.id = i
            user.served = self.train_K
            self.users.append(user)
        # Add dummy users
        for i in range(N+1, self.train_N+1):
            user = User(mode=self.mode)
            user.id = i
            user.alpha = 2
            user.beta = 2
            user.served = self.train_K
            self.users.append(user)

        # 2) Fill in pickup/dropoff coords, windows
        for i in range(1, 2*N + 1):
            node = instance['instance'][i+1]  # i+1 => skip the depot line in the instance file
            # If i> N => dropoff
            # else => pickup
            # Adjust indexing if needed
            if i > N and N < self.train_N:
                i_adj = i + (self.train_N - N)
            else:
                i_adj = i
            
            user_id = i_adj if i_adj <= N else (i_adj - N)
            user = self.users[user_id - 1]

            if i <= N:
                # pickup
                user.pickup_coords = [float(node[1]), float(node[2])]
                user.serve_duration = node[3]
                user.load = node[4]
                user.pickup_window = [float(node[5]), float(node[6])]
            else:
                # dropoff
                user.dropoff_coords = [float(node[1]), float(node[2])]
                user.dropoff_window = [float(node[5]), float(node[6])]

        # 3) Time-window tightening
        for user in self.users:
            if user.id > N:
                continue
            travel_time = euclidean_distance(user.pickup_coords, user.dropoff_coords)
            # If user.id <= N/2 => drop-off requests? (This logic depends on how you label them.)
            # Or if your data is random, adapt as needed.
            if user.id <= (N/2):
                user.pickup_window[0] = round(max(0.0, user.dropoff_window[0] - L - user.serve_duration),3)
                user.pickup_window[1] = round(min(user.dropoff_window[1] - travel_time - user.serve_duration, T),3)
            else:
                user.dropoff_window[0] = round(max(0.0, user.pickup_window[0] + user.serve_duration + travel_time),3)
                user.dropoff_window[1] = round(min(user.pickup_window[1] + user.serve_duration + L, T),3)

        # 4) Vehicles
        self.vehicles = []
        for k in range(K):
            v = Vehicle(mode=self.mode)
            v.id = k
            v.route = instance['routes'][k]
            v.route.insert(0, 0)
            v.route.append(2*self.train_N + 1)
            v.schedule = instance['schedule'][k]
            v.free_capacity = Q
            self.vehicles.append(v)
        for k in range(K, self.train_K):
            v = Vehicle(mode=self.mode)
            v.id = k
            v.free_time = 1440
            self.vehicles.append(v)

        if self.mode != 'supervise':
            self.break_window = []
            self.break_ride_time = []
            self.break_same = []
            self.break_done = []
            self.time_penalty = 0

        # 5) Arc elimination (build self.arcs dict)
        self.arc_elimination(display=True)

        # 6) Initialize or re-initialize the graph
        if not self.graph_initialized:
            self.init_graph()   # Build the base structure, nodes, edges
            self.graph_initialized = True
        # We always do a full re-check of edges so they match the brand-new instance state
        self.update_edge_feasibility()

        return instance['objective']
    
    def arc_elimination(self, display=False):
        """
        Computes the arcs dict with pairs of node ids as key and True/False as value. 
        Useful for eliminating some of the edges in the graph.
        Node ids follow the same names as in (Cordeau 2006 and Dumas 1991).
        A conversion from our node ids is required to use it.
        """
        # self.eliminated_arcs = []
        _, N, _, Q, L = self.parameter()

        self.nodes = []
        self.arcs = {}
        for i in range(2 * N + 2):
            for j in range(2 * N + 2):
                if i != j:
                    self.arcs[(i, j)] = True

        # Do not perform arc elimination if not requested
        if not self.args.arc_elimination:
            return

        # Pick-up sources
        for i in range(N):
            node = Node()
            node.coords = self.users[i].pickup_coords
            node.serve_duration = self.users[i].serve_duration
            node.load = self.users[i].load
            node.window = self.users[i].pickup_window
            self.nodes.append(node)
        # Drop-off destinations
        for i in range(N):
            node = Node()
            node.coords = self.users[i].dropoff_coords
            node.serve_duration = self.users[i].serve_duration
            node.load = - self.users[i].load
            node.window = self.users[i].dropoff_window
            self.nodes.append(node)
        # Source and destination station
        source = Node()
        destination = Node()
        # Time-window tightening (Section 5.1.1, Cordeau 2006)
        tightening_e = []
        tightening_l = []
        for i in range(2 * N):
            node = self.nodes[i]
            tightening_e.append(node.window[0] - euclidean_distance(source.coords, node.coords))
            tightening_l.append(
                node.window[1] + node.serve_duration + euclidean_distance(node.coords, destination.coords))
        source.window[0], source.window[1] = min(tightening_e), max(tightening_l)
        destination.window[0], destination.window[1] = source.window[0], source.window[1]
        self.nodes.insert(0, source)
        self.nodes.append(destination)

        # Basic arc elimination (Cordeau 2006 and Dumas 1991)
        num_eliminated_arcs = [0]
        num_remaining_arcs = [sum([arc[1] for arc in list(self.arcs.items())])]
        if display:
            print("Step 0: # eliminated: {:.0f} -> # remaining: {:.0f}"
                  .format(num_eliminated_arcs[-1], num_remaining_arcs[-1]))

        # Priority and Pairing
        for i in range(1, N + 1):
            # Priority
            self.arcs[(0, N + i)] = False
            self.arcs[(N + i, i)] = False
            self.arcs[(2 * N + 1, 0)] = False
            self.arcs[(2 * N + 1, i)] = False
            self.arcs[(2 * N + 1, N + i)] = False
            # Pairing
            self.arcs[(i, 2 * N + 1)] = False
        if display:
            num_eliminated_arcs.append(num_remaining_arcs[-1] - sum([arc[1] for arc in list(self.arcs.items())]))
            num_remaining_arcs.append(num_remaining_arcs[-1] - num_eliminated_arcs[-1])
            print("Step 1: # eliminated: {:.0f} -> # remaining: {:.0f}"
                  .format(num_eliminated_arcs[-1], num_remaining_arcs[-1]))

        # Vehicle capacity
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i != j:
                    node_i = self.nodes[i]
                    node_j = self.nodes[j]
                    if node_i.load + node_j.load > Q:
                        self.arcs[(i, j)] = False
                        self.arcs[(j, i)] = False
                        self.arcs[(i, N + j)] = False
                        self.arcs[(j, N + i)] = False
                        self.arcs[(N + i, N + j)] = False
                        self.arcs[(N + j, N + i)] = False
        if display:
            num_eliminated_arcs.append(num_remaining_arcs[-1] - sum([arc[1] for arc in list(self.arcs.items())]))
            num_remaining_arcs.append(num_remaining_arcs[-1] - num_eliminated_arcs[-1])
            print("Step 2: # eliminated: {:.0f} -> # remaining: {:.0f}"
                  .format(num_eliminated_arcs[-1], num_remaining_arcs[-1]))
        
        # Time windows
        for i in range(0, 2 * N + 2):
            for j in range(0, 2 * N + 2):
                if i != j:
                    node_i = self.nodes[i]
                    node_j = self.nodes[j]
                    travel_time = euclidean_distance(node_i.coords, node_j.coords)
                    if node_i.window[0] + node_i.serve_duration + travel_time > node_j.window[1] + 1e-3:
                        self.arcs[(i, j)] = False
        if display:
            num_eliminated_arcs.append(num_remaining_arcs[-1] - sum([arc[1] for arc in list(self.arcs.items())]))
            num_remaining_arcs.append(num_remaining_arcs[-1] - num_eliminated_arcs[-1])
            print("Step 3: # eliminated: {:.0f} -> # remaining: {:.0f}"
                  .format(num_eliminated_arcs[-1], num_remaining_arcs[-1]))

        # Time windows and pairing of requests
        for i in range(1, N + 1):
            for j in range(0, 2 * N + 2):
                if N + i != j and i != j:
                    node_i = self.nodes[i]
                    node_n = self.nodes[N + i]
                    node_j = self.nodes[j]
                    travel_time_ij = euclidean_distance(node_i.coords, node_j.coords)
                    travel_time_jn = euclidean_distance(node_j.coords, node_n.coords)
                    if travel_time_ij + node_j.serve_duration + travel_time_jn > L:
                        self.arcs[(i, j)] = False
                        self.arcs[(j, N + i)] = False
        if display:
            num_eliminated_arcs.append(num_remaining_arcs[-1] - sum([arc[1] for arc in list(self.arcs.items())]))
            num_remaining_arcs.append(num_remaining_arcs[-1] - num_eliminated_arcs[-1])
            print("Step 4: # eliminated: {:.0f} -> # remaining: {:.0f}"
                  .format(num_eliminated_arcs[-1], num_remaining_arcs[-1]))
        

            entries = [num_remaining_arcs[0]] + num_eliminated_arcs[1:] + [num_remaining_arcs[-1]]
            print("&", entries[0], "&", entries[1], "&", entries[2], "&", entries[3], "&", entries[4], "&", entries[5])

    def is_arc_feasible(self, i_u, i_v):
        """
        i_u and i_v are indices of nodes as in the state graph.
        Returns True if the edge between u and v is feasible, i.e the corresponding value is True in the arcs dictionary.
        """
        # Converted indices to (Cordeau 2006 and Dumas 1991) indices
        converted_i_u = i_u
        converted_i_v = i_v
        if i_u == 0 or i_v == 0:
            # Everything is connected to the waiting node
            return True
        if i_u >= self.num_nodes - self.train_K:
            # Source node
            converted_i_u = 0
        if i_v >= self.num_nodes - self.train_K:
            # Source node
            converted_i_v = 0

        if converted_i_u == converted_i_v:
            return False
        return self.arcs[(converted_i_u, converted_i_v)]
        

    def beta(self, k):
        for i in range(0, self.train_N):
            user = self.users[i]
            if user.alpha == 1 and user.served == self.vehicles[k].id:
                # 1: the user is being served by the vehicle performing an action at time step t
                user.beta = 1
            else:
                if user.alpha == 0:
                    if user.load <= self.vehicles[k].free_capacity:
                        # 0: the user is waiting to be served
                        user.beta = 0
                    else:
                        # 2: the user cannot be served by the vehicle
                        user.beta = 2
                else:
                    # 2: the user has been served
                    user.beta = 2
    


    ########################################################################
    #                   Graph Construction (single graph)                  #
    ########################################################################

    def init_graph(self):
        """
        Build a single DGL graph with all possible nodes & edges, store in self.graph.
        We'll store node info in self.node_info so we can do is_edge checks each step.
        """
        K, N = self.train_K, self.train_N
        n_nodes = 2*N + K + 2  # 0..(2N+K+1)
        n_features = 17
        self.node_features = torch.zeros(n_nodes, n_features, device=self.device)

        # Build node_info
        self.node_info = []
        # Node 0 -> wait
        self.node_info.append({
            'index': 0, 'type': 'wait',
            'user': None, 'vehicle': None
        })
        # Pickup nodes -> 1..N
        for i in range(1, N+1):
            self.node_info.append({
                'index': i,
                'type': 'pickup',
                'user': self.users[i-1],
                'vehicle': None
            })
        # Dropoff nodes -> N+1..2N
        for i in range(1, N+1):
            self.node_info.append({
                'index': N+i,
                'type': 'dropoff',
                'user': self.users[i-1],
                'vehicle': None
            })
        # Destination -> 2N+1
        self.node_info.append({
            'index': 2*N+1,
            'type': 'destination',
            'user': None,
            'vehicle': None
        })
        # Source nodes -> 2N+2..2N+K+1
        for k_i in range(K):
            self.node_info.append({
                'index': 2*N+2 + k_i,
                'type': 'source',
                'user': None,
                'vehicle': self.vehicles[k_i]
            })

        # Create graph
        g = dgl.graph(([],[]), num_nodes=n_nodes, device=self.device)
        g.ndata['feat'] = self.node_features

        # Add edges for all possible pairs (i_u != i_v). We'll filter feasibility in edge features.
        edges_src = []
        edges_dst = []
        edges_feat = []  # shape (#edges, 5) e.g. [dist, pairing, waiting, feasible, reverse_feasible]

        for u_info in self.node_info:
            for v_info in self.node_info:
                if u_info['index']!=v_info['index']:
                    edges_src.append(u_info['index'])
                    edges_dst.append(v_info['index'])
                    edges_feat.append([0.0, 0, 0, 1, 1])  # placeholder

        g.add_edges(edges_src, edges_dst)
        g.edata['feat'] = torch.tensor(edges_feat, device=self.device)
        self.graph = g

    def update_edge_feasibility(self):
        """
        Re-check is_edge(...) for every edge. Also update the edge feats: dist, pairing, waiting, feasible bits.
        """

        edges_src, edges_dst = self.graph.edges()
        edges_feat = self.graph.edata['feat']  # shape (#E, 5)
        # We'll define them as: 
        #   0 -> distance
        #   1 -> pairing
        #   2 -> waiting
        #   3 -> feasible
        #   4 -> reverse_feasible

        for e_id in range(self.graph.num_edges()):
            i_u = edges_src[e_id].item()
            i_v = edges_dst[e_id].item()

            u_info = self.node_info[i_u]
            v_info = self.node_info[i_v]

            # fetch user or vehicle
            u_user = u_info['user']
            v_user = v_info['user']
            k_u = u_info['vehicle']
            k_v = v_info['vehicle']
            t_u = u_info['type']
            t_v = v_info['type']

            # next vehicle logic if needed:
            u_next = False
            v_next = False
            # (optionally set these if you need them)

            # Distance
            dist = 0.0
            # For demonstration, let’s do a simple approach:
            if u_user and (t_u=='pickup'):
                start_coord = u_user.pickup_coords
            elif u_user and (t_u=='dropoff'):
                start_coord = u_user.dropoff_coords
            elif k_u:
                start_coord = k_u.coords
            else:
                start_coord = [0.0, 0.0]

            if v_user and (t_v=='pickup'):
                end_coord = v_user.pickup_coords
            elif v_user and (t_v=='dropoff'):
                end_coord = v_user.dropoff_coords
            elif k_v:
                end_coord = k_v.coords
            else:
                end_coord = [0.0, 0.0]

            dist = euclidean_distance(start_coord, end_coord)

            # Pairing = 1 if same user, else 0
            pairing = 1 if (u_user and v_user and u_user is v_user) else 0

            # waiting = 1 if either is wait
            waiting = 1 if (t_u=='wait' or t_v=='wait') else 0

            feasible = 1 if is_edge(self, i_u, u_user, k_u, t_u, u_next,
                                          i_v, v_user, k_v, t_v, v_next) else 0
            reverse_feasible = 1 if is_edge(self, i_v, v_user, k_v, t_v, v_next,
                                                 i_u, u_user, k_u, t_u, u_next) else 0

            edges_feat[e_id, 0] = dist
            edges_feat[e_id, 1] = pairing
            edges_feat[e_id, 2] = waiting
            edges_feat[e_id, 3] = feasible
            edges_feat[e_id, 4] = reverse_feasible

        self.graph.edata['feat'] = edges_feat





    ########################################################################
    #                   state_graph() for backward compat                  #
    ########################################################################

    def state_graph(self, k, current_time):
        """
        For old code that calls `state_graph(k, time)`,
        we return (graph, next_vehicle_node).

        We will:
          1) Update node features for the current time/vehicle k
          2) update_edge_feasibility()
          3) Find next_vehicle_node (like old code).
          4) Return (self.graph, next_vehicle_node).
        """
        # 1) update node features (like old code: pickups, dropoffs, wait node, source node)
        self.update_node_features(k, current_time)
        # 2) update edges
        self.update_edge_feasibility()
        # 3) find which node is “next_vehicle_node”
        next_vehicle_node = self.find_vehicle_node(k)

        # 4) return
        return self.graph, next_vehicle_node

    def update_node_features(self, k, current_time):
        """
        Re-zero self.node_features. Then fill in features for each node
        (pickup, dropoff, wait, source, destination).
        You can replicate your old approach if needed.
        """
        node_features = self.graph.ndata['feat']
        node_features.zero_()

        K, N, T, Q, L = self.parameter()

        # Example logic: loop over self.node_info
        for info in self.node_info:
            i_nd = info['index']
            t_nd = info['type']
            v_user = info['user']
            v_veh = info['vehicle']

            # set one-hot
            node_features[i_nd, one_hot_node_type(t_nd)] = 1

            # if pickup or dropoff, store coords and windows
            if v_user and t_nd=='pickup':
                # shift the window by current_time if you want
                w_start, w_end = shift_window(v_user.pickup_window, current_time)
                node_features[i_nd, 5] = v_user.pickup_coords[0]
                node_features[i_nd, 6] = v_user.pickup_coords[1]
                node_features[i_nd, 7] = w_start
                node_features[i_nd, 8] = w_end
                node_features[i_nd, 9] = v_user.serve_duration
                node_features[i_nd, 10] = L - v_user.ride_time
                node_features[i_nd, 11] = v_user.load

                # check if a vehicle is present
                if (v_veh := self.vehicle_present(v_user.pickup_coords)):
                    node_features[i_nd, 12] = 1
                    node_features[i_nd, 13] = v_veh.free_capacity
                    node_features[i_nd, 14] = v_veh.free_time
                    node_features[i_nd, 15] = T
                    if k == v_veh.id:
                        node_features[i_nd, 16] = 1
                        #print(f"Vehicle {k} is at pickup node {i_nd}")


            elif v_user and t_nd=='dropoff':
                w_start, w_end = shift_window(v_user.dropoff_window, current_time)
                node_features[i_nd, 5] = v_user.dropoff_coords[0]
                node_features[i_nd, 6] = v_user.dropoff_coords[1]
                node_features[i_nd, 7] = w_start
                node_features[i_nd, 8] = w_end
                node_features[i_nd, 9] = v_user.serve_duration
                node_features[i_nd, 10] = L - v_user.ride_time
                node_features[i_nd, 11] = -v_user.load

                if (v_veh := self.vehicle_present(v_user.dropoff_coords)):
                    node_features[i_nd, 12] = 1
                    node_features[i_nd, 13] = v_veh.free_capacity
                    node_features[i_nd, 14] = v_veh.free_time
                    node_features[i_nd, 15] = T
                    if k == v_veh.id:
                        node_features[i_nd, 16] = 1

            elif t_nd=='wait':
                # put the wait node at the vehicle k's coords
                if k < len(self.vehicles):
                    node_features[i_nd, 5] = self.vehicles[k].coords[0]
                    node_features[i_nd, 6] = self.vehicles[k].coords[1]

            elif t_nd=='source' and v_veh is not None:
                # if vehicle is still at depot, mark it
                if v_veh.coords == [0.0,0.0] and v_veh.free_time<1440:
                    node_features[i_nd, 12] = 1
                    node_features[i_nd, 13] = v_veh.free_capacity
                    node_features[i_nd, 14] = v_veh.free_time
                    node_features[i_nd, 15] = T
                    if k == v_veh.id:
                        node_features[i_nd, 16] = 1

        self.graph.ndata['feat'] = node_features

    def find_vehicle_node(self, k):
        """
        Decide which node index is "occupied" or "controlled" by vehicle k, 
        to return as next_vehicle_node. 
        For example, if the vehicle is at a pickup node i, we return i. 
        If it's at wait=0, we return 0, etc.
        """
        # We'll do a simple pass over node_info and see if we set node_features[i,16] = 1
        # Then that i is the next_vehicle_node.

        node_feats = self.graph.ndata['feat']
        # The "is next available" bit is index 16
        print(self.graph.num_nodes())
        for i_nd in range(self.graph.num_nodes()):
            if node_feats[i_nd, 16] ==1 :
                return i_nd
        # fallback
        return 0

    def vehicle_present(self, coords):
        """
        Return a vehicle if its coords match the given coords (exact or close).
        """
        for v in self.vehicles:
            # watch out for float precision
            if v.coords == coords:
                return v
        return None



    def action2node(self, action):
        """
        Given an action, returns the corresponding node in the state graph as a tensor.
        """
        if action < self.train_N:  # user
            if self.users[action].alpha == 0:
                return torch.tensor(action + 1, device=self.device)  # pickup
            else:
                return torch.tensor((action + 1) + self.train_N, device=self.device)  # dropoff
        elif action == self.train_N:  # destination
            return torch.tensor(2 * action + 1, device=self.device)  # 2N + 1
        elif action == self.train_N + 1:  # wait
            return torch.tensor(0, device=self.device)
        else:

            raise RuntimeError('Action not recognized')

    
    # def node2action(self, node):
    #     """
    #     given an node in the state graph, returns the corresponding action
    #     """
    
    #     if node == 0: # wait
    #         return torch.tensor(self.train_N + 1, device= self.device)
    #     elif node <= 2*self.train_N: # user
    #         return torch.tensor((node - 1) % self.train_N, device=self.device)
    #     elif node == 2*self.train_N + 1: # destination
    #         return torch.tensor(self.train_N, device= self.device)
    #     else:
    #         print(node)
    #         raise RuntimeError('Action not recognized')
    
    def node2action(self, node):
        if node == 0:  # wait
            return torch.tensor(self.train_N + 1, device=self.device)
        elif node <= 2*self.train_N:  # user
            return torch.tensor((node-1) % self.train_N, device=self.device)
        elif node == 2*self.train_N + 1:  # destination
            return torch.tensor(self.train_N, device=self.device)
        elif node >= 2*self.train_N + 2 and node <= 2*self.train_N + self.train_K + 1:
            # Source node => treat it as wait or define a new action
            return torch.tensor(self.train_N + 1, device=self.device)
        else:
            raise RuntimeError(f'Action not recognized. Node = {node}')

    # noinspection PyMethodMayBeStatic
    def will_pick_up(self, vehicle, user):
        """ Called when a vehicle arrives at a pickup location """
        travel_time = euclidean_distance(vehicle.coords, user.pickup_coords)
        window_start = user.pickup_window[0]
        vehicle.coords = user.pickup_coords
        vehicle.free_capacity -= user.load
        user.served = vehicle.id
        user.alpha = 1

        return travel_time, window_start

    def will_drop_off(self, vehicle, user):
        """ Called when a vehicle arrives at a dropoff location """
        travel_time = euclidean_distance(vehicle.coords, user.dropoff_coords)
        window_start = user.dropoff_window[0]
        vehicle.coords = user.dropoff_coords
        vehicle.free_capacity += user.load
        user.served = self.train_K
        user.alpha = 2

        return travel_time, window_start

    def action(self, k):
        """ Computes action taken by the expert policy """
        vehicle = self.vehicles[k]
        r = vehicle.ordinal
        node = vehicle.route[r]
        isDropOff = node > self.train_N and node <= 2*self.train_N # we do not allow to wait on dropoff nodes
        if vehicle.free_time + self.args.wait_time < vehicle.schedule[r] and not isDropOff:
            # Wait at the present node
            action = self.train_N + 1
        else:
            if vehicle.route[r + 1] < 2 * self.train_N + 1:
                # Move to the next node
                node = vehicle.route[r + 1]
                action = self.node2user[node] - 1
            else:
                # Move to the destination depot
                action = self.train_N

        return action

    def supervise_step(self, k):
        """
        Simulate a step of the instance to reach the next state.
        Action taken by the expert, dataset creation time.
        Return the travel time of the transition.
        """

        vehicle = self.vehicles[k]
        r = vehicle.ordinal

        if vehicle.free_time + self.args.wait_time < vehicle.schedule[r]:
            # Wait at the present node
            vehicle.free_time += self.args.wait_time
            update_ride_time(vehicle, self.users, self.args.wait_time)
            travel_time = 0.0
        else:
            wait_time = vehicle.schedule[r] - vehicle.free_time
            update_ride_time(vehicle, self.users, wait_time)
            vehicle.free_time = vehicle.schedule[r]

            if vehicle.route[r] != 0:
                # Start to serve the user at the present node
                node = vehicle.route[r]
                user = self.users[self.node2user[node] - 1]

                if user.id not in vehicle.serving:
                    # Check the pick-up time window
                    if check_window(user.pickup_window, vehicle.free_time):
                        raise ValueError('The pick-up time window of User {} is broken: {:.2f} not in {}.'.format(
                            user.id, vehicle.free_time, user.pickup_window))
                    # Append the user to the serving list
                    vehicle.serving.append(user.id)
                else:
                    # Check the ride time
                    if user.ride_time - user.serve_duration > self.train_L + 1e-2:
                        raise ValueError('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                            user.id, user.ride_time - user.serve_duration, self.train_L))
                    # Check the drop-off time window
                    if check_window(user.dropoff_window, vehicle.free_time):
                        raise ValueError('The drop-off time window of User {} is broken: {:.2f} not in {}.'.format(
                            user.id, vehicle.free_time, user.dropoff_window))
                    # Remove the user from the serving list
                    vehicle.serving.remove(user.id)

                vehicle.serve_duration = user.serve_duration
                user.ride_time = 0.0

            if vehicle.route[r + 1] < 2 * self.train_N + 1:
                # Move to the next node
                node = vehicle.route[r + 1]
                user = self.users[self.node2user[node] - 1]

                if user.id not in vehicle.serving:
                    travel_time, window_start = self.will_pick_up(vehicle, user)
                else:
                    travel_time, window_start = self.will_drop_off(vehicle, user)

                if vehicle.free_time + vehicle.serve_duration + travel_time > window_start + 1e-2:
                    ride_time = vehicle.serve_duration + travel_time
                    vehicle.free_time += ride_time
                else:
                    ride_time = window_start - vehicle.free_time
                    vehicle.free_time = window_start

                update_ride_time(vehicle, self.users, ride_time)
            else:
                # Move to the destination depot
                travel_time = euclidean_distance(vehicle.coords, [0.0, 0.0])
                vehicle.coords = [0.0, 0.0]
                vehicle.free_time = 1440
                vehicle.serve_duration = 0

            vehicle.ordinal += 1

        return travel_time

    def predict(self, vehicle_node_id, user_mask=None, src_mask=None):
        graph =self.graph.to(self.device)
        ks = torch.tensor([vehicle_node_id], device=self.device)
        batch_x = graph.ndata['feat'].to(self.device)
        batch_e = graph.edata['feat'].to(self.device)
        
        policy_outputs, value_outputs = self.model(graph, batch_x, batch_e, ks, self.num_nodes,masking=True)  #h_lap_pe=batch_lap_pe

        probs = f.softmax(policy_outputs, dim=1)
        _, action_node = torch.max(probs, 1)

        # value outputs to also be returned in the future
        return action_node.item(), probs

    def evaluate_step(self, k, action):
        """
        Simulate a step of the instance to reach the next state.
        Action taken by the model, at test time.
        Return the travel time of the transition.
        """
        K, N, T, Q, L = self.parameter()
        vehicle = self.vehicles[k]

        constraint_penalty = 0.0

        if action == self.train_N + 1:
            # Wait at the present node
            vehicle.free_time += self.args.wait_time
            update_ride_time(vehicle, self.users, self.args.wait_time)
            travel_time = 0.0
        else:
            if vehicle.pred_route[-1] != 0:
                # Start to serve the user at the present node
                user = self.users[vehicle.pred_route[-1] - 1]

                if user.id not in vehicle.serving:
                    # Check the pick-up time window
                    if check_window(user.pickup_window, vehicle.free_time) and user.id > N / 2:
                        if self.logs:
                            print('The pick-up time window of User {} is broken: {:.2f} not in {}.'.format(
                                user.id, vehicle.free_time, user.pickup_window))
                        self.break_window.append(user.id)
                        self.time_penalty += vehicle.free_time - user.pickup_window[1]
                        constraint_penalty += vehicle.free_time - user.pickup_window[1]
                    # Append the user to the serving list
                    vehicle.serving.append(user.id)
                else:
                    # Check the ride time
                    if user.ride_time - user.serve_duration > L + 1e-2:
                        if self.logs:
                            print('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                                user.id, user.ride_time - user.serve_duration, L))
                        self.break_ride_time.append(user.id)
                        self.time_penalty += user.ride_time - user.serve_duration - L
                        constraint_penalty += user.ride_time - user.serve_duration - L
                    # Check the drop-off time window
                    if check_window(user.dropoff_window, vehicle.free_time) and user.id <= N / 2:
                        if self.logs:
                            print('The drop-off time window of User {} is broken: {:.2f} not in {}.'.format(
                                user.id, vehicle.free_time, user.dropoff_window))
                        self.break_window.append(user.id)
                        self.time_penalty += vehicle.free_time - user.dropoff_window[1]
                        constraint_penalty += vehicle.free_time - user.dropoff_window[1]
                    # Remove the user from the serving list
                    vehicle.serving.remove(user.id)

                vehicle.serve_duration = user.serve_duration
                user.ride_time = 0.0

            if action < N:
                # Move to the next node
                user = self.users[action]

                if user.id not in vehicle.serving:
                    travel_time, window_start = self.will_pick_up(vehicle, user)
                    user.pred_served.append(vehicle.id)
                else:
                    travel_time, window_start = self.will_drop_off(vehicle, user)
                    user.pred_served.append(vehicle.id)

                if vehicle.free_time + vehicle.serve_duration + travel_time > window_start + 1e-2:
                    ride_time = vehicle.serve_duration + travel_time
                    vehicle.free_time += ride_time
                else:
                    ride_time = window_start - vehicle.free_time
                    vehicle.free_time = window_start

                vehicle.pred_cost += travel_time
                update_ride_time(vehicle, self.users, ride_time)
            else:
                # Move to the destination depot
                travel_time = euclidean_distance(vehicle.coords, [0.0, 0.0])
                vehicle.pred_cost += travel_time
                vehicle.coords = [0.0, 0.0]
                vehicle.free_time = 1440
                vehicle.serve_duration = 0

            vehicle.pred_route.append(action + 1)
            vehicle.pred_schedule.append(vehicle.free_time)

        return travel_time, constraint_penalty

    def finish(self):
        """ Returns True if the DARP is not done, False if the episode is over. """
        free_times = np.array([vehicle.free_time for vehicle in self.vehicles])
        num_finish = np.sum(free_times >= 1440)

        if num_finish == self.train_K:
            flag = False
            if self.mode != 'supervise':
                _, N, _, _, _ = self.parameter()

                for i in range(0, N):
                    user = self.users[i]
                    # Check if the user is served by the same vehicle.
                    if len(user.pred_served) != 2 or user.pred_served[0] != user.pred_served[1]:
                        self.break_same.append(user.id)
                        print('* User {} is served by {}.'.format(user.id, user.pred_served))
                    # Check if the request of the user is finished.
                    if user.alpha != 2:
                        self.break_done.append(user.id)
                        print('* The request of User {} is unfinished.'.format(user.id))
        else:
            flag = True
            self.graph_initialized = False

        return flag

    def cost(self):
        return sum(vehicle.pred_cost for vehicle in self.vehicles)

    def parameter(self):
        if self.mode != 'evaluate':
            return self.train_K, self.train_N, self.train_T, self.train_Q, self.train_L
        else:
            return self.test_K, self.test_N, self.test_T, self.test_Q, self.test_L
        