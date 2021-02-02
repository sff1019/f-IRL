import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np


class ResNetAIRLDisc(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layer_blocks=2,
                 hid_dim=100,
                 hid_act='relu',
                 use_bn=True,
                 clamp_magnitude=10.0):
        # TODO: add LayerNorm https://pytorch.org/docs/master/generated/torch.nn.LayerNorm.html
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude
        self.input_dim = input_dim

        self.first_fc = nn.Linear(input_dim, hid_dim)

        self.blocks_list = nn.ModuleList()
        for i in range(num_layer_blocks - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hid_dim, hid_dim))
            if use_bn: block.append(nn.BatchNorm1d(hid_dim))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        self.blocks_list = self.blocks_list

        self.last_fc = nn.Linear(hid_dim, 1)

    def forward(self, batch):
        x = self.first_fc(batch)
        for block in self.blocks_list:
            x = x + block(x)  # residual network
        output = self.last_fc(x)
        if self.clamp_magnitude is not None:
            output = torch.clamp(output,
                                 min=-1.0 * self.clamp_magnitude,
                                 max=self.clamp_magnitude)
        return output


class SMMIRLDisc(nn.Module):
    # for log density ratio estimation in f-div
    def __init__(self,
                 input_dim,
                 num_layer_blocks=2,
                 hid_dim=100,
                 hid_act='relu',
                 use_bn=True,
                 clamp_magnitude=10.0,
                 lr=0.0003,
                 weight_decay=0.0001,
                 momentum=0.9,
                 batch_size=128,
                 device=torch.device('cpu'),
                 **kwargs):
        super().__init__()

        self.model = ResNetAIRLDisc(input_dim, num_layer_blocks, hid_dim,
                                    hid_act, use_bn,
                                    clamp_magnitude).to(device)

        self.input_dim = input_dim
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    betas=(momentum, 0.999))

        # https://pytorch.org/docs/master/generated/torch.nn.BCEWithLogitsLoss.html
        self.bce = nn.BCEWithLogitsLoss().to(device)
        self.bce_targets = torch.cat(
            [torch.ones(batch_size, 1),
             torch.zeros(batch_size, 1)], dim=0).to(device)
        self.batch_size = batch_size

    def log_density_ratio(self, states):
        states = torch.FloatTensor(states).to(self.device)
        self.eval()
        with torch.no_grad():
            logits = self.model.forward(states)
            # before sigmoid. i.e. output = log(D(s)) - log(1-D(s)) where D(s) is the original Discriminator
            # NOTE: for optimal discriminator, return log(density ratio)
        self.train()
        return logits.squeeze(1)

    def learn(self, expert_samples, agent_samples, iter: int):
        expert_samples = torch.FloatTensor(expert_samples).to(self.device)
        agent_samples = torch.FloatTensor(agent_samples).to(self.device)
        assert expert_samples.shape[1] == agent_samples.shape[
            1] == self.input_dim
        expert_numbers, agent_numbers = expert_samples.shape[
            0], agent_samples.shape[0]
        disc_loss = []

        for i in range(iter):
            self.optimizer.zero_grad()

            expert_batch = expert_samples[np.random.choice(
                expert_numbers, size=self.batch_size)].clone()
            agent_batch = agent_samples[np.random.choice(
                agent_numbers, size=self.batch_size)].clone()

            expert_logits = self.model.forward(expert_batch)
            agent_logits = self.model.forward(agent_batch)

            disc_logits = torch.cat([expert_logits, agent_logits], dim=0)
            disc_ce_loss = self.bce(disc_logits, self.bce_targets)
            # print(i, disc_ce_loss.item())
            # TODO: may add grad pen

            disc_ce_loss.backward()
            self.optimizer.step()
            disc_loss.append(disc_ce_loss.item())

        return disc_loss
