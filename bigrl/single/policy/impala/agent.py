import torch
from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from bigrl.single.import_helper import import_pipeline_module
from collections import deque


class BaseAgent:
    HAS_MODEL = True

    def __init__(self, cfg=None):
        self.whole_cfg = cfg
        self.player_id = self.whole_cfg.agent.player_id
        self.env_name = self.whole_cfg.env.name
        self.pipeline = self.whole_cfg.agent.pipeline
        self.setup_model()
        self.send_data = self.whole_cfg.agent.send_data
        self.model_last_iter = torch.zeros(size=())
        self.reset()

    def reset(self, ):
        # self.hidden_state = hidden_state()
        if self.send_data:
            self.data_buffer = deque(maxlen=self.whole_cfg.learner.data.unroll_len)
            self.push_count = 0
        self.agent_iter = 0

    def setup_model(self):
        self.ModelClass = import_pipeline_module(self.env_name, self.pipeline, 'Model', )
        self.model = self.ModelClass(self.whole_cfg,)

    def preprocess(self, obs):
        '''
        Args:
            obs:
                original obs
        Returns:
            model input: Dict of logits, hidden states, action_log_probs, action_info
            value_feature[Optional]: Dict of global info
        '''
        obs = torch.from_numpy(obs).unsqueeze(0)
        return obs

    def step(self, obs):
        self.model_input = obs
        if self.send_data:
            with torch.no_grad():
                self.model_output = self.model.compute_logp_action(self.model_input)
        else:
            with torch.no_grad():
                self.model_output = self.model.forward(self.model_input, temperature=0)
        action = self.model_output['action'].squeeze(0).detach().numpy()
        return action

    def eval_postprocess(self, next_obs):
        # self.hidden_state = self.model_output["hidden_state"]
        pass

    def collect_data(self, next_obs, reward, done, ):
        step_data = {
            'obs': self.model_input,
            # 'hidden_state': self.hidden_state,
            'action': self.model_output['action'],
            'action_logp': self.model_output['action_logp'],
            'reward': torch.tensor([reward]),
            'done': torch.tensor([done]),
            'model_last_iter': torch.tensor([self.model_last_iter.item()], dtype=torch.float),
        }
        # push data
        self.data_buffer.append(step_data)
        self.push_count += 1
        # self.hidden_state = self.model_output["hidden_state"]

        if self.push_count == self.whole_cfg.learner.data.unroll_len:
            last_step_data = {
                'obs': self.preprocess(next_obs),
                # 'hidden_state': self.hidden_state,
            }
            list_data = list(self.data_buffer)
            list_data.append(last_step_data)
            self.push_count = 0
            return_data = default_collate_with_dim(list_data,cat=True )

            return_data = return_data
        else:
            return_data = None

        return return_data
