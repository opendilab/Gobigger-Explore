import torch

from bigrl.single.worker.learner.rl_learner import BaseRLLearner


class RLLearner(BaseRLLearner):
    def setup_teacher_model(self):
        self.use_teacher = self.whole_cfg.agent.get('use_teacher', True)
        if self.use_teacher:
            self.teacher_checkpoint_path = self.whole_cfg.agent.teacher_checkpoint_path
            self.teacher_model = self.ModelClass(self.whole_cfg, use_value_network=False)
            state_dict = torch.load(self.teacher_checkpoint_path, map_location='cpu')
            model_state_dict = {k: v for k, v in state_dict['model'].items() if
                                'value' not in k}
            self.teacher_model.load_state_dict(model_state_dict, strict=False)
            if self.use_cuda:
                self.teacher_model = self.teacher_model.to(device=self.device)
            self.logger.info(f'load teacher model from:{self.teacher_checkpoint_path}')

    def setup_model(self):
        super(RLLearner, self).setup_model()
        self.setup_teacher_model()

    def model_forward(self, data):
        with self.timer:
            with torch.enable_grad():
                data = self.model.rl_train(data)
            if self.use_teacher:
                with torch.no_grad():
                    data = self.teacher_model.teacher_forward(data)
            total_loss, loss_info_dict = self.loss.compute_loss(data)
        forward_time = self.timer.value
        self.variable_record.update_var({'forward': forward_time})
        self.loss_record.update_var({'forward': forward_time})
        self.loss_record.update_var(loss_info_dict)
        return total_loss
