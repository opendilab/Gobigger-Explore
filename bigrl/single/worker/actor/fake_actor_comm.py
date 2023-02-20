from copy import deepcopy


class FakeActorComm:
    def __init__(self, local_job):
        self.job = local_job
        self.active_model_dict = {}
        self.send_data_players = {}

    def ask_for_job(self, ):
        return deepcopy(self.job)

    def send_data(self, data, player_id,):
        pass

    def send_result(self, result_info):
        pass

    def close(self):
        pass

    def update_model(self):
        pass
