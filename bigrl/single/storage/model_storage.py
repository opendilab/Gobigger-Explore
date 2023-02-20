import torch
from bigrl.core.torch_utils.data_helper import to_device


class ModelStorage:
    def __init__(self,strict=False):
        self.model_dict = {}
        self.last_iter_dict = {}
        self.strict=strict

    def push(self, player_id, model_class,cfg, last_iter=0):
        model = model_class(cfg).to(device='cpu').eval().share_memory()
        self.model_dict[player_id] = model
        self.last_iter_dict[player_id] = torch.tensor([last_iter],dtype=torch.long).share_memory_()

    def pull(self, player_id):
        if player_id not in self.model_dict or player_id not in self.last_iter_dict:
            return None, None
        return self.model_dict[player_id], self.last_iter_dict[player_id]

    def update(self,player_id,model, last_iter):
        model_state_dict = to_device({k: v for k, v in model.state_dict().items()}, device='cpu')
        self.model_dict[player_id].load_state_dict(model_state_dict, strict=self.strict)
        self.last_iter_dict[player_id].copy_(torch.tensor([last_iter],dtype=torch.long))
