import torch
import torch.nn as nn
import torch.nn.functional as F


class Mixer(nn.Module):
    """
    Overview:
        mixer network in QMIX, which mix up the independent q_value of each agent to a total q_value
    Interface:
        __init__, forward
    """

    def __init__(self, cfg):
        """
        Overview:
            initialize pymarl mixer network
        Arguments:
            - n_agents (:obj:`int`): the number of agent
            - state_dim(:obj:`int`): the dimension of global observation state
            - mixing_embed_dim (:obj:`int`): the dimension of mixing state emdedding
            - hypernet_embed (:obj:`int`): the dimension of hypernet emdedding, default to 64
        """
        super(Mixer, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.mixer
        
        self.n_agents = self.cfg.n_agents
        self.state_dim = self.cfg.state_dim
        self.embed_dim = self.cfg.mixing_embed_dim
        
        hypernet_embed = 64
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim * self.n_agents)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed), nn.ReLU(), nn.Linear(hypernet_embed, self.embed_dim)
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        """
        Overview:
            forward computation graph of pymarl mixer network
        Arguments:
            - agent_qs (:obj:`torch.FloatTensor`): the independent q_value of each agent
            - states (:obj:`torch.FloatTensor`): the emdedding vector of global state
        Returns:
            - q_tot (:obj:`torch.FloatTensor`): the total mixed q_value
        Shapes:
            - agent_qs (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is agent_num
            - states (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is embedding_size
            - q_tot (:obj:`torch.FloatTensor`): :math:`(B, )`
        """
        bs = agent_qs.shape[:-1]
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(*bs)
        return q_tot