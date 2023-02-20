from copy import deepcopy
import torch
from collections import defaultdict
import time
import random
import os
from easydict import EasyDict
from bigrl.single.import_helper import import_pipeline_agent
from bigrl.single.worker.actor.actor import BaseActor
from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from .env.env import make_env


class Actor(BaseActor):

    def _inference_loop(self, env_id, comm, variable_record,):
        torch.set_num_threads(1)
        time.sleep(env_id * 10 * random.random())
        job_remaining_use_count = 0
        ask_job_period = max(self.whole_cfg.communication.get('actor_ask_job_period', 10), 1)
        episode = 0
        ori_job = None
        last_log_show_time = time.time()
        while True:
            # ask for job
            if job_remaining_use_count > 0 and ori_job is not None:
                job, merged_whole_cfg = self.get_job_config(ori_job,episode, env_id=env_id)
                job_remaining_use_count -= 1
            else:
                with self.timer:
                    ori_job = comm.ask_for_job()
                variable_record.update_var({'ask_for_job': self.timer.value})
                job_remaining_use_count = ask_job_period - 1
                job, merged_whole_cfg = self.get_job_config(ori_job,episode, env_id=env_id)
                actor_agents = []
                historic_models = {}
                for idx, (player_id, pipeline,config_path) in enumerate(
                        zip(job['player_id'], job['pipeline'],job['config_path'])):
                    Agent = import_pipeline_agent(self.env_name, pipeline, 'Agent')
                    if config_path and os.path.exists(config_path):
                        agent_config = read_config(config_path)
                    else:
                        agent_config = {}
                    merged_agent_cfg = deep_merge_dicts(self.whole_cfg, agent_config)
                    merged_agent_cfg.agent.player_id = player_id
                    merged_agent_cfg.agent.pipeline = pipeline
                    merged_agent_cfg.agent.send_data = self.job_type == 'train' and player_id in job[
                        'send_data_players']
                    merged_agent_cfg.agent.game_player_id = idx
                    agent = Agent(merged_agent_cfg)
                    if agent.HAS_MODEL:
                        if player_id in comm.model_storage.model_dict:
                            agent.model = comm.model_storage.model_dict[player_id]
                            agent.model_last_iter = comm.model_storage.last_iter_dict[player_id]
                        elif player_id in historic_models:
                            agent.model = historic_models[player_id]
                        else:
                            agent.setup_model()

                            if os.path.exists(job['checkpoint_path'][idx]):
                                state_dict = torch.load(job['checkpoint_path'][idx], map_location='cpu')
                                model_state_dict = {k: v for k, v in state_dict['model'].items() if
                                                'value' not in k}
                                agent.model.load_state_dict(model_state_dict, strict=False)
                            historic_models[player_id] = agent.model
                    actor_agents.append(agent)

            env = make_env(merged_whole_cfg.env,)
            obs = env.reset()
            for agent in actor_agents:
                agent.reset()

            # env episode loop
            while True:
                # show log
                if env_id == 0 and time.time() - last_log_show_time > 10:
                    info_text = f"\n{'=' * 5}Actor-{self.actor_uid}{'=' * 5}\n{variable_record.get_vars_text()}\n"
                    self.logger.info(info_text)
                    last_log_show_time = time.time()
                # agent step
                actions = {}
                for agent in actor_agents:
                    with self.timer:
                        action = agent.step(obs)
                    variable_record.update_var({'agent_step': self.timer.value})
                    actions.update(action)

                with self.timer:
                    next_obs, reward, done, info = env.step(actions)
                variable_record.update_var({'env_step': self.timer.value})

                # collect data
                for agent in actor_agents:
                    if agent.send_data:
                        with self.timer:
                            traj_data = agent.collect_data(next_obs, reward, done, info)
                        variable_record.update_var({'collect_data': self.timer.value})
                        if traj_data is not None:
                            with self.timer:
                                comm.send_data(traj_data, player_id=agent.player_id, )
                            variable_record.update_var({'send_data': self.timer.value})
                            self.send_data_counter.copy_(self.send_data_counter + 1)
                    else:
                        with self.timer:
                            agent.eval_postprocess(next_obs, reward, done, info)
                        variable_record.update_var({'eval_postprocess': self.timer.value})

                if not done:
                    obs = next_obs
                else:
                    players_stat = defaultdict(lambda: defaultdict(list))
                    leader_board = list(next_obs[0]['leaderboard'].values())
                    sorted_leader_board = sorted(leader_board, reverse=True)
                    for agent in actor_agents:
                        players_stat[agent.player_id]['score'].append(next_obs[1][agent.game_player_id]['score'])
                        players_stat[agent.player_id]['team_score'].append(next_obs[0]['leaderboard'][agent.game_team_id])
                        players_stat[agent.player_id]['rank'].append(
                            sorted_leader_board.index(leader_board[agent.game_team_id]))
                        if hasattr(agent, 'stat'):
                            for k, v in agent.stat.items():
                                players_stat[agent.player_id][k].append(v)
                    for k, v in info['eats'].items():
                        player_id = actor_agents[k].player_id
                        for _k, _v in v.items():
                            players_stat[player_id][_k].append(_v)

                    result_info = {}
                    for player_id, v in players_stat.items():
                        player_stat = {}
                        for _k, _v in v.items():
                            if _k == 'score':
                                max_score = max(_v)
                                player_stat['max_score'] = max_score
                            if _k == 'team_score':
                                max_score = max(_v)
                                player_stat['max_team_score'] = max_score
                            avg = sum(_v) / len(_v)
                            player_stat[_k] = avg
                        if self.whole_cfg.get('league').get('eval_sl', False):
                            if 'bot' in players_stat.keys():
                                player_stat['score_bot'] = player_stat['score']
                            else:
                                player_stat['score_sp'] = player_stat['score']
                        result_info[player_id] = player_stat
                    result_info['branch'] = job['branch']
                    comm.send_result(result_info)

                    score = {k: v['score'] for k, v in players_stat.items()}
                    print(f'episode: {episode}, env_id{env_id} done, with info:{score}', flush=True)
                    episode += 1
                    break

            env.close()
        return True

    def get_job_config(self, ori_job,episode, env_id):
        # init env and actor agents
        save_replay_freq = self.cfg.get('save_replay_freq',0)
        job = deepcopy(ori_job)
        env_info = job.get('env_info', {})
        merged_whole_cfg = deep_merge_dicts(self.whole_cfg, {'env': env_info})
        player_num_per_team = self.whole_cfg.env.player_num_per_team
        replay_name = ''.join([p for idx, p in enumerate(job['player_id']) if idx % player_num_per_team == 0])
        if 'playback_settings' not in merged_whole_cfg.env:
            merged_whole_cfg.env['playback_settings '] = EasyDict({})
        # env_id == 0 will save replay every save_replay_freq  episode
        if env_id == 0 and not self.cfg.get('debug_mode', False) and self.job_type == 'train' \
                and save_replay_freq > 0 and episode % save_replay_freq == 0:
            default_save_dir = os.path.join('replays', self.whole_cfg.common.experiment_name)
            playback_type = merged_whole_cfg.env.playback_settings.get('playback_type', '')
            if playback_type == 'by_frame':
                merged_whole_cfg.env.playback_settings.by_frame.save_frame = True
                replay_dir = merged_whole_cfg.env.playback_settings.by_frame.get('save_dir', default_save_dir)
                save_dir = os.path.join(replay_dir, replay_name)
                merged_whole_cfg.env.playback_settings.by_frame.save_dir = save_dir

            elif playback_type == 'by_video':
                merged_whole_cfg.env.playback_settings.by_video.save_video = True
                replay_dir = merged_whole_cfg.env.playback_settings.by_video.get('save_dir', default_save_dir)
                save_dir = os.path.join(replay_dir, replay_name)
                merged_whole_cfg.env.playback_settings.by_video.save_dir = save_dir

        return job, merged_whole_cfg