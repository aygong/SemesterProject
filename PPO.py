import random
import torch
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
from environment import *
from utils import *
from evaluation import *
from graph_transformer import GraphTransformerNet
import time

class PPO:
    def __init__(
            self,
            args,
            device,
            model_name,
            gamma,
            lr,
            clip_rate,
            value_loss_coef,
            batch_size,
            n_epochs,
            collect_episodes,
            update_epochs,
            num_instances,
            save_rate,
            path_result,
            entropy_coef = 1e-3,
            entropy_coef_decay = 0.99,
            constraint_penalty_alpha = 10.0
            ):
        
        self.args = args
        self.device = device
        self.model_name = model_name
        self.gamma = gamma
        self.lr = lr
        self.clip_rate = clip_rate
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.value_loss_coef = value_loss_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.collect_episodes = collect_episodes
        self.update_epochs = update_epochs
        self.num_instances = num_instances
        self.save_rate = save_rate
        self.path_result = path_result
        self.constraint_penalty_alpha = constraint_penalty_alpha
        
        # Episodes data
        self.states = []
        self.action_nodes = []
        self.costs = []
        self.log_probs = []
        self.dones = []
        self.values = []
        self.vehicle_node_ids = []

        self.darp = Darp(args, mode='reinforce', device=device)
        self.darp.model = GraphTransformerNet(
            device=device,
            num_nodes=2*self.darp.train_N + self.darp.train_K + 2,
            num_node_feat=17,
            num_edge_feat=3,
            d_model=128,
            num_layers=4,
            num_heads=8,
            dropout=0.1
        )
        
        checkpoint = torch.load('./model/sl-' + model_name + '.model')
        self.darp.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.darp.model.to(device)
        self.model = self.darp.model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion_value = torch.nn.MSELoss()

    def greedy_action(self, graph, vehicle_node_id):
        # Select action with max probability
        graph = graph.to(self.device)
        k = torch.tensor([vehicle_node_id], device=self.device)
        x = graph.ndata['feat'].to(self.device)
        e = graph.edata['feat'].to(self.device)

        with torch.no_grad():
            policy, value = self.model(graph, x, e, k, masking=True)
            probs = f.softmax(policy, dim=1)
            a = torch.argmax(probs).item()
            p = probs[a].item()
        return a, p, value
    
    def select_action(self, graph, vehicle_node_id):
        # Stochastic action selection for one state
        graph = graph.to(self.device)
        k = torch.tensor([vehicle_node_id], device=self.device)
        x = graph.ndata['feat'].to(self.device)
        e = graph.edata['feat'].to(self.device)

        with torch.no_grad():
            policy, value = self.model(graph, x, e, k, masking=True)
            probs = f.softmax(policy, dim=1)
            cat = Categorical(probs=probs)

            action = cat.sample()
            
        return action, cat.log_prob(action), value
    
    def evaluate_policy(self, states, vehicle_node_ids, action_nodes):
        # Policy evaluation for a batch of states
        # vehicle_node_ids and action_nodes are given as tensors on desired device
        # states is a list of graphs

        batched_graph = dgl.batch(states).to(self.device)
        
        batch_x = batched_graph.ndata['feat'].to(self.device)
        batch_e = batched_graph.edata['feat'].to(self.device)

        policy_outputs, value_outputs = self.model(batched_graph, batch_x, batch_e, vehicle_node_ids, masking=True)
        probs = f.softmax(policy_outputs, dim=1)
        cat = Categorical(probs=probs)

        return cat.log_prob(action_nodes), cat.entropy(), value_outputs
    
    def get_state(self):
        free_times = [vehicle.free_time for vehicle in self.darp.vehicles]
        time = np.min(free_times)
        indices = np.argwhere(free_times == time)
        indices = indices.flatten().tolist()

        k = indices[0]
        
        if self.darp.vehicles[k].free_time >= 1440:
            raise RuntimeError(f'Darp whould be finished, free_times: {free_times}')
            
        self.darp.beta(k)
        state, next_vehicle_node = self.darp.state_graph(k, time)

        return state, k, next_vehicle_node

    
    def collect_data(self):
        # Collect data for collect_episodes episodes.
        # Data = s, a, r, s_prime, pi_a, done
        rl_instances = list(range(self.args.num_rl_instances))
        random.shuffle(rl_instances)
        rl_instances_iter = iter(rl_instances)

        for _ in range(self.collect_episodes):
            # Run one episode
            num_instance = next(rl_instances_iter)
            objective = self.darp.reset(num_instance)

            done = not self.darp.finish()

            episode_n_actions = 0

            while not done:
                state, k, next_vehicle_node = self.get_state()

                action_node, log_prob, current_value = self.select_action(state, next_vehicle_node)
                action = self.darp.node2action(action_node)
                travel_time, constraint_penalty = self.darp.evaluate_step(k, action)
                transition_cost = travel_time 
                done = not self.darp.finish()

                if done:
                    # give large penalty for users that are not served correctly
                    transition_cost += 500 * (len(self.darp.break_done) + len(self.darp.break_same))

                # Give penalty to previous actions, not current one
                # Current one is leaving the user whose window was broken, so we need to penalize two actions before
                # (action that did something else than coming to this user)
                penalty_index = min(episode_n_actions, 2)
                if penalty_index == 0:
                    transition_cost += self.constraint_penalty_alpha * constraint_penalty
                else:
                    self.costs[-penalty_index] += self.constraint_penalty_alpha * constraint_penalty

                episode_n_actions += 1
                
                    
                # Store data
                self.states.append(state)
                self.action_nodes.append(action_node)
                self.costs.append(transition_cost)
                self.log_probs.append(log_prob)
                self.dones.append(done)
                self.values.append(current_value)
                self.vehicle_node_ids.append(next_vehicle_node)

    
    def compute_returns(self):
        # Compute returns of trajectories
        self.returns = np.zeros(len(self.costs))
        with torch.no_grad():
            for t in reversed(range(len(self.costs))):
                if t == len(self.costs) - 1:
                    next_return = 0
                else:
                    next_return = self.returns[t+1]
                nextnonterminal = 0 if self.dones[t] else 1
                self.returns[t] = self.costs[t] + nextnonterminal * next_return
            self.advantages = (np.array(self.values) - self.returns) /(np.array(self.values)) # the smaller the returns (cost) the better
            
    
    

    def data_to_tensors(self):
        # Create torch tensors from collected data.
        states, action_nodes, costs, log_probs, dones, values, vehicle_node_ids, returns, advantages = shuffle_list(self.states,
                                                                                                  self.action_nodes,
                                                                                                  self.costs,
                                                                                                  self.log_probs,
                                                                                                  self.dones,
                                                                                                  self.values,
                                                                                                  self.vehicle_node_ids,
                                                                                                  self.returns,
                                                                                                  self.advantages)
        with torch.no_grad():
            action_nodes_tensor = torch.tensor(action_nodes, device=self.device).long()
            costs_tensor = torch.tensor(costs, device=self.device)
            log_probs_tensor = torch.tensor(log_probs, device=self.device)
            dones_tensor = torch.tensor(dones, device=self.device)
            values_tensor = torch.tensor(values, device=self.device)
            vehicle_node_tensor = torch.tensor(vehicle_node_ids, device=self.device)
            returns_tensor = torch.tensor(returns, device=self.device)
            advantages_tensor = torch.tensor(advantages, device=self.device)
        
        return states, action_nodes_tensor, costs_tensor, log_probs_tensor, dones_tensor, values_tensor, vehicle_node_tensor, returns_tensor, advantages_tensor
    
    def clear_data(self):
        self.states = []
        self.action_nodes = []
        self.costs = []
        self.log_probs = []
        self.dones = []
        self.values = []
        self.vehicle_node_ids = []
        self.returns = None
        self.advantages = None

    def update_model(self):
        # Update model parameters with collected data

        policy_losses = np.zeros(self.update_epochs)
        value_losses = np.zeros(self.update_epochs)
        entropy_list = np.zeros(self.update_epochs)

        for epoch in range(self.update_epochs):
            states, action_nodes, costs, log_probs, dones, values, vehicle_node_ids, returns, advantages = self.data_to_tensors()

            for start in range(0, len(costs), self.batch_size):
                
                end = start + self.batch_size
                if start >= len(costs)-1:
                   # 0-d tensors
                   break

                new_log_probs, entropies, new_values = self.evaluate_policy(states[start:end], vehicle_node_ids[start:end], action_nodes[start:end])
                log_ratio = new_log_probs - log_probs[start:end]
                ratio = log_ratio.exp() # a/b == exp(log(a)-log(b))
                
                policy_loss1 = -advantages[start:end] * ratio
                policy_loss2 = -advantages[start:end] * torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                #value_loss = self.criterion_value(new_values / returns[start:end], torch.ones(len(new_values)).to(self.device))
                value_loss = ((new_values / returns[start:end] - torch.ones(len(new_values)).to(self.device))**2).mean()

                entropy = entropies.mean()

                loss = policy_loss - self.entropy_coef * entropy + value_loss * self.value_loss_coef

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                policy_losses[epoch] += policy_loss.item()
                value_losses[epoch] += value_loss.item()
                entropy_list[epoch] += entropy.item()
            
            policy_losses[epoch] /= len(costs)//self.batch_size
            value_losses[epoch] /= len(costs)//self.batch_size
            entropy_list[epoch] /= len(costs)//self.batch_size

        return policy_losses, value_losses, entropy_list
    
    def train(self):
        # Train policy and value networks on collected data
        self.model.train()
        
        # Loop:
        for epoch in range(self.n_epochs):
            start_time = time.time()
            # collect data
            print('----- Collect data -----')
            self.clear_data()
            self.collect_data()

            # compute returns
            print('----- Compute returns -----')
            self.compute_returns()

            # update model
            print('----- Update model -----')
            policy_losses, value_losses, entropy_list = self.update_model()
            self.entropy_coef *= self.entropy_coef_decay # less exploration for later times
            # save model
            if epoch % self.save_rate == 0:
                self.save(epoch, self.model_name)
                evaluation(self.args, self.model)
            
            # print results
            exec_time = time.time() - start_time
            print(f'epoch: {epoch}, execution time: {exec_time / 3600}, policy loss: {np.mean(policy_losses)}, value loss: {np.mean(value_losses)}, entropy: {np.mean(entropy_list)}')
            with open(self.path_result + 'PPO_log.txt', 'a+') as file:
                json.dump({
                    'epoch': epoch,
                    'execution time': exec_time / 3600,
                    'policy loss': np.mean(policy_losses),
                    'value loss': np.mean(value_losses),
                    'entropy': np.mean(entropy_list),
                }, file)
                file.write("\n")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            #'scheduler_state_dict': self.scheduler.state_dict(),
        }, './model/' + 'rl-' + self.model_name + '.model')
        
        return self.model
    
    def save(self, episode, model_name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            #'scheduler_state_dict': self.scheduler.state_dict(),
        }, "./model/ppo-{}-{}.model".format(model_name, episode))
    


