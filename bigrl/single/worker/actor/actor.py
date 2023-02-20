# -*- coding:utf-8 -*


import time
from copy import deepcopy

import torch
from bigrl.single.import_helper import import_pipeline_agent, import_env_module

from bigrl.single.worker.actor.base_actor import BaseActor


class Actor(BaseActor):

    def _inference_loop(self, env_id, comm, variable_record, ):
        torch.set_num_threads(1)
        last_log_show_time = time.time()
        with self.timer:
            job = comm.ask_for_job()
        variable_record.update_var({'ask_for_job': self.timer.value})
        make_env = import_env_module(env_name=self.whole_cfg.env.name, name='make_env')
        env = make_env(self.whole_cfg)

        # actor log
        pipeline = job['pipeline'][0]
        player_id = job['player_id'][0]

        Agent = import_pipeline_agent(self.env_name, pipeline, 'Agent')
        agent_cfg = deepcopy(self.whole_cfg)
        agent_cfg.agent.player_id = player_id
        agent_cfg.agent.pipeline = pipeline
        agent_cfg.agent.send_data = self.job_type == 'train' and player_id in job['send_data_players']
        agent = Agent(agent_cfg)

        if player_id in comm.model_storage.model_dict:
            agent.model = comm.model_storage.model_dict[player_id]
            agent.model_last_iter = comm.model_storage.last_iter_dict[player_id]
        else:
            state_dict = torch.load(job['checkpoint_path'][0], map_location='cpu')
            model_state_dict = {k: v for k, v in state_dict['model'].items() if
                                'value' not in k}
            agent.model.load_state_dict(model_state_dict, strict=False)

        obs = env.reset()
        agent.reset()
        count = 0

        while True:
            # show log
            if env_id == 0 and time.time() - last_log_show_time > self.whole_cfg.actor.log_show_freq:
                info_text = f"\n{'=' * 5}Actor-{self.actor_uid}{'=' * 5}\n{variable_record.get_vars_text()}\n"
                self.logger.info(info_text)
                last_log_show_time = time.time()
            # env loop
            while True:
                # agent step
                with self.timer:
                    agent_obs = agent.preprocess(obs)
                variable_record.update_var({'preprocess': self.timer.value})
                with self.timer:
                    action = agent.step(agent_obs)
                variable_record.update_var({'agent_step': self.timer.value})

                with self.timer:
                    next_obs, reward, done, info = env.step(action)
                    count += 1
                variable_record.update_var({'env_step': self.timer.value})

                # collect data
                if agent.send_data:
                    with self.timer:
                        traj_data = agent.collect_data(next_obs, reward, done)
                    variable_record.update_var({'collect_data': self.timer.value})
                    if traj_data is not None:
                        with self.timer:
                            comm.send_data(traj_data, player_id=agent.player_id, )
                        variable_record.update_var({'send_data': self.timer.value})
                        self.send_data_counter.copy_(self.send_data_counter + 1)

                else:
                    with self.timer:
                        agent.eval_postprocess(next_obs)
                    variable_record.update_var({'eval_postprocess': self.timer.value})

                if not done:
                    obs = next_obs
                else:
                    self.logger.info(f'env_id{env_id} done, with info:{info}', )
                    if 'eval' in self.job_type:
                        player_stat = {'eval_reward': info['cumulative_rewards']}
                    else:
                        player_stat = {'reward': info['cumulative_rewards']}
                    result_info = {
                        'player_id': agent.player_id,
                        'player_stat': player_stat,
                    }
                    comm.send_result(result_info)
                    obs = env.reset()
                    break
        env.close()

        return True

