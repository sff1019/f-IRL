import numpy as np
import torch
import torch.nn.functional as F


def f_div_loss(div: str, IS: bool, samples, rho_expert, agent_density,
               reward_func, device):
    # please add eps to expert density, not here
    s, _, log_a = samples
    N, T, d = s.shape

    s_vec = s.reshape(-1, d)
    log_density_ratio = np.log(
        rho_expert(s_vec)) - agent_density.score_samples(s_vec).reshape(-1)
    log_density_ratio = torch.FloatTensor(log_density_ratio).to(device)

    t1 = torch.exp(log_density_ratio)  # (N*T,) p/q TODO: clip

    t1 = (-t1).view(N, T).sum(1)  # NOTE: sign (N,)
    t2 = reward_func.r(torch.FloatTensor(s_vec).to(device)).view(N, T).sum(
        1)  # (N,)

    if IS:
        traj_reward = reward_func.get_scalar_reward(s_vec).reshape(N, T).sum(
            1)  # (N,)
        traj_log_prob = log_a.sum(1)  # (N,)
        IS_ratio = F.softmax(torch.FloatTensor(traj_reward - traj_log_prob),
                             dim=0).to(device)  # normalized weight
        surrogate_objective = (IS_ratio * t1 * t2).sum() - (
            IS_ratio * t1).sum() * (IS_ratio * t2).sum()
    else:
        surrogate_objective = (
            t1 * t2).mean() - t1.mean() * t2.mean()  # sample covariance

    surrogate_objective /= T
    return surrogate_objective, t1 / T  # log of geometric mean w.r.t. traj (0 is the borderline)
