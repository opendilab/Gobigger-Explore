import logging

from flask import Flask, request

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def create_league_app(league):
    app = Flask(__name__)

    def build_ret(code, info=''):
        return {'code': code, 'info': info}

    # ************************** learner *********************************
    @app.route('/league/register_learner', methods=['POST'])
    def register_learner():
        ret_info = league.deal_with_register_learner(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    @app.route('/league/learner_send_train_info', methods=['POST'])
    def send_train_info():
        ret_info = league.deal_with_learner_send_train_info(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    # ************************** actor *********************************
    @app.route('/league/register_actor', methods=['POST'])
    def register_actor():
        ret_info = league.deal_with_register_actor(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    @app.route('/league/actor_ask_for_job', methods=['POST'])
    def ask_for_job():
        ret_info = league.deal_with_actor_ask_for_job(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    @app.route('/league/actor_send_result', methods=['POST'])
    def send_result():
        ret_info = league.deal_with_actor_send_result(request.json)
        if ret_info:
            return build_ret(0)
        else:
            return build_ret(1)

    # *********
    # debug use
    # *********
    # ************************** resume *********************************
    @app.route('/league/save_resume', methods=['GET'])
    def save_resume():
        resume_path = league.save_resume()
        return {'resume_path': resume_path}

    @app.route('/league/load_resume', methods=['POST'])
    def load_resume():
        ret_info = league.deal_with_load_resume(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    # ************************** sl_eval *********************************
    @app.route('/league/save_sl_tb', methods=['GET'])
    def save_sl_tb():
        league.reset_sl_tb()
        sl_tb_log = league.save_sl_tb()
        return {'sl_tb_path': sl_tb_log}

    # ************************** config *********************************
    @app.route('/league/show_config', methods=['GET'])
    def show_config():
        league.save_config(print_config=True)
        return league.whole_cfg

    @app.route('/league/update_config', methods=['GET'])
    def update_cfg():
        ret_info = league.update_config()
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    # ************************** player *********************************
    @app.route('/league/display_player', methods=['POST'])
    def display_player():
        ret_info = league.display_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/add_active_player', methods=['POST'])
    def add_active_player():
        ret_info = league.deal_with_add_active_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/add_hist_player', methods=['POST'])
    def add_hist_player():
        ret_info = league.deal_with_add_hist_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/update_player', methods=['POST'])
    def update_player():
        ret_info = league.deal_with_update_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/refresh_players', methods=['GET'])
    def refresh_players():
        league.deal_with_refresh_players()
        return {'Done': 'successfully refresh_all_players'}

    @app.route('/league/remove_player', methods=['POST'])
    def remove_player():
        ret_info = league.deal_with_remove_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/reset_player', methods=['POST'])
    def reset_player_stat():
        ret_info = league.deal_with_reset_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    # ************************** ladder *********************************
    # followings apis are be used when we use trueskill/elo in league
    @app.route('/league/show_elo', methods=['GET'])
    def show_elo():
        if hasattr(league,'show_elo'):
            ret_info = league.show_elo()
            if ret_info:
                return build_ret(0, )
            else:
                return build_ret(1)
        else:
            return build_ret(1,info='NotImplemented')

    @app.route('/league/refresh_elo', methods=['GET'])
    def refresh_elo():
        if hasattr(league,'refresh_elo'):
            ret_info = league.refresh_elo()
            if ret_info:
                return build_ret(0, )
            else:
                return build_ret(1)
        else:
            return build_ret(1,info='NotImplemented')

    @app.route('/league/show_trueskill', methods=['GET'])
    def show_trueskill():
        if hasattr(league,'show_trueskill'):
            ret_info = league.show_trueskill()
            if ret_info:
                return build_ret(0, )
            else:
                return build_ret(1)
        else:
            return build_ret(1,info='NotImplemented')

    @app.route('/league/refresh_trueskill', methods=['GET'])
    def refresh_trueskill():
        if hasattr(league,'refresh_trueskill'):
            ret_info = league.refresh_trueskill()
            if ret_info:
                return build_ret(0, )
            else:
                return build_ret(1)
        else:
            return build_ret(1,info='NotImplemented')
    return app
