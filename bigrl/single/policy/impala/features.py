import torch
from bigrl.core.torch_utils.collate_fn import default_collate_with_dim

class Features:
    def __init__(self, cfg={}):
        self.cfg = cfg

    def get_rl_step_data(self,last=False):
        data = {}
        data['obs'] = torch.zeros(size=(1, 4, 84, 84), dtype=torch.float)
        if not last:
            data['action'] = torch.zeros(size=(1,),dtype=torch.long)
            data['action_logp'] = torch.zeros(size=(1,),dtype=torch.float)
            data['reward'] = torch.zeros(size=(1,),dtype=torch.float)
            data['done'] = torch.zeros(size=(1,),dtype=torch.bool)
            data['model_last_iter'] = torch.zeros(size=(1,),dtype=torch.float)
        return data

    def get_rl_traj_data(self,unroll_len):
        traj_data_list = []
        for _ in range(unroll_len):
            traj_data_list.append(self.get_rl_step_data())
        traj_data_list.append(self.get_rl_step_data(last=True))
        traj_data = default_collate_with_dim(traj_data_list, cat=True)
        return traj_data

    def get_rl_batch_data(self,unroll_len,batch_size):
        batch_data_list = []
        for _ in range(batch_size):
            batch_data_list.append(self.get_rl_traj_data(unroll_len))
        batch_data = default_collate_with_dim(batch_data_list,dim=1)
        return batch_data

if __name__ == '__main__':
    features = Features()
    traj_data = features.get_rl_traj_data(unroll_len=10)
    for k,val in traj_data.items():
        print(k,val.shape)

    batch_data = features.get_rl_batch_data(unroll_len=10,batch_size=3)
    for k, val in  batch_data.items():
        print(k,val.shape)
