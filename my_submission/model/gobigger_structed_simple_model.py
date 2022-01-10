import torch
import torch.nn as nn
from ding.torch_utils import MLP, get_lstm, Transformer
from ding.model import DiscreteHead
from ding.utils import list_split

class RelationGCN(nn.Module):

    def __init__(
            self,
            hidden_shape: int,
            activation=nn.ReLU(inplace=True),
    ) -> None:
        super(RelationGCN, self).__init__()
        # activation
        self.act = activation
        # layers
        self.thorn_relation_layers = MLP(
            hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )
        self.clone_relation_layers = MLP(
            hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )
        self.agg_relation_layers = MLP(
            4 * hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )

    def forward(self, food_relation, thorn_relation, clone, clone_relation, thorn_mask, clone_mask):
        b, t, c = clone.shape[0], thorn_relation.shape[2], clone.shape[1]
        # encode thorn relation
        thorn_relation = self.thorn_relation_layers(thorn_relation) * thorn_mask.view(b, 1, t, 1)  # [b,n_clone,n_thorn,c]
        thorn_relation = thorn_relation.max(2).values # [b,n_clone,c]
        # encode clone relation
        clone_relation = self.clone_relation_layers(clone_relation) * clone_mask.view(b, 1, c, 1) # [b,n_clone,n_clone,c]
        clone_relation = clone_relation.max(2).values # [b,n_clone,c]
        # encode aggregated relation
        agg_relation = torch.cat([clone, food_relation, thorn_relation, clone_relation], dim=2)
        clone = self.agg_relation_layers(agg_relation)
        return clone

class Encoder(nn.Module):
    def __init__(
            self,
            scalar_shape: int,
            food_shape: int,
            food_relation_shape: int,
            thorn_relation_shape: int,
            clone_shape: int,
            clone_relation_shape: int,
            hidden_shape: int,
            encode_shape: int,
            activation=nn.ReLU(inplace=True),
    ) -> None:
        super(Encoder, self).__init__()

        # scalar encoder
        self.scalar_encoder = MLP(
            scalar_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # food encoder
        layers = []
        kernel_size = [5, 3, 1]
        stride = [4, 2, 1]
        shape = [hidden_shape // 4, hidden_shape // 2, hidden_shape]
        input_shape = food_shape
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_shape, shape[i], kernel_size[i], stride[i], kernel_size[i] // 2))
            layers.append(activation)
            input_shape = shape[i]
        self.food_encoder = nn.Sequential(*layers)
        # food relation encoder
        self.food_relation_encoder = MLP(
            food_relation_shape, hidden_shape // 2, hidden_shape, layer_num=2, activation=activation
        )
        # thorn relation encoder
        self.thorn_relation_encoder = MLP(
            thorn_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone encoder
        self.clone_encoder = MLP(
            clone_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone relation encoder
        self.clone_relation_encoder = MLP(
            clone_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # gcn
        self.gcn = RelationGCN(
            hidden_shape, activation=activation
        )
        self.agg_encoder = MLP(
            3 * hidden_shape, hidden_shape, encode_shape, layer_num=2, activation=activation
        )
    
    def forward(self, scalar, food, food_relation, thorn_relation, thorn_mask, clone, clone_relation, clone_mask):
        # encode scalar
        scalar = self.scalar_encoder(scalar) # [b,c]
        # encode food
        food = self.food_encoder(food) # [b,c,h,w]
        food = food.reshape(*food.shape[:2], -1).max(-1).values # [b,c]
        # encode food relation
        food_relation = self.food_relation_encoder(food_relation) # [b,c]
        # encode thorn relation
        thorn_relation = self.thorn_relation_encoder(thorn_relation) # [b,n_clone,n_thorn, c]
        # encode clone
        clone = self.clone_encoder(clone) # [b,n_clone,c]
        # encode clone relation
        clone_relation = self.clone_relation_encoder(clone_relation) # [b,n_clone,n_clone,c]
        # aggregate all relation
        clone = self.gcn(food_relation, thorn_relation, clone, clone_relation, thorn_mask, clone_mask)
        clone = clone.max(1).values # [b,c]

        return self.agg_encoder(torch.cat([scalar, food, clone], dim=1))

class GoBiggerHybridActionSimpleV3(nn.Module):
    r"""
    Overview:
        The GoBiggerHybridAction model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_encoder``, ``compute_critic``
    """
    def __init__(
            self,
            scalar_shape: int,
            food_shape: int,
            food_relation_shape: int,
            thorn_relation_shape: int,
            clone_shape: int,
            clone_relation_shape: int,
            hidden_shape: int,
            encode_shape: int,
            action_type_shape: int,
            rnn: bool = False,
            activation=nn.ReLU(inplace=True),
    ) -> None:
        super(GoBiggerHybridActionSimpleV3, self).__init__()
        self.activation = activation
        self.action_type_shape = action_type_shape
        # encoder
        self.encoder = Encoder(scalar_shape, food_shape, food_relation_shape, thorn_relation_shape, clone_shape, clone_relation_shape, hidden_shape, encode_shape, activation)
        # head
        self.action_type_head = DiscreteHead(32, action_type_shape, layer_num=2, activation=self.activation)

    def forward(self, inputs):
        scalar = inputs['scalar']
        food = inputs['food']
        food_relation = inputs['food_relation']
        thorn_relation = inputs['thorn_relation']
        thorn_mask = inputs['thorn_mask']
        clone = inputs['clone']
        clone_relation = inputs['clone_relation']
        clone_mask = inputs['clone_mask']
        fused_embedding_total = self.encoder(scalar, food, food_relation, thorn_relation, thorn_mask, clone, clone_relation, clone_mask)
        B = inputs['batch']
        A = inputs['player_num_per_team']

        action_type_logit = self.action_type_head(fused_embedding_total)['logit']  # B, M, action_type_size
        action_type_logit = action_type_logit.reshape(B, A, *action_type_logit.shape[1:])

        result = {
            'logit': action_type_logit,
        }
        return result