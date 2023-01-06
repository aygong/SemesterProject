import torch
from transformer import Transformer
from environment import Darp
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from evaluation import greedy_evaluation
import time

def reinforce(args):
    cuda_available = torch.cuda.is_available()
    device = get_device(cuda_available)

    darp = Darp(args, mode='evaluate', device=device)

    # Create a model
    darp.model = Transformer(
        device=device,
        num_vehicles=darp.train_K,
        input_seq_len=darp.train_N,
        target_seq_len=darp.train_N + 2,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        dropout=args.dropout)

    # Load the trained model
    model_name = darp.train_name + '-' + str(args.wait_time)
    checkpoint = torch.load('./model/model-' + model_name + '.model')
    darp.model.load_state_dict(checkpoint['model_state_dict'])
    darp.model.train()
    darp.logs = False

    if cuda_available:
        darp.model.cuda()

    #load optimizer and scheduler
    optimizer = torch.optim.Adam(darp.model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.99)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    run_times = []

    src_mask = [1 for _ in range(darp.test_N)] + [0 for _ in range(darp.train_N - darp.test_N)]
    # user_mask = torch.Tensor(user_mask).to(device)
    src_mask = torch.Tensor(src_mask).to(device)

    for instance in range(args.num_rl_instances):
        start = time.time()
        objective = darp.reset(instance)
        cost = greedy_evaluation(darp, instance, src_mask)

        sum_log_probs = sum(darp.log_probs)
        undelivered = sum([user.served == 0 for user in darp.users])
        train_R = undelivered * 100.0  + darp.time_penalty + max((cost - objective)/objective * 1000 , 0)

        loss = torch.mul(-train_R, sum_log_probs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(darp.model.parameters(), 0.5)
        optimizer.step()

        running_loss += loss.item()
        scheduler.step(running_loss)
        
        run_time = time.time() - start
        run_times.append(run_time)

        print('Instance {}, run time {}, cost: {}, objective {}, time penalty {}'.format \
                (instance, round(run_time, 4), round(cost, 4), round(objective, 4), round(darp.time_penalty, 4)))

    print('Training finished.')
    print('Average execution time per instance: {:.4f} seconds.'.format(sum(run_times) / len(run_times)))
    print("Total execution time: {:.4f} seconds.".format(sum(run_times))) 

    torch.save({
            'model_state_dict': darp.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, './model/' + 'reinforce-' + model_name + '.model')

       



