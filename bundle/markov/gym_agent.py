import os
import zmq
import json
import msgpack
import msgpack_numpy as m
from rl_coach.core_types import ActionInfo
from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgent


m.patch()

AGENT_PARAMS_PATH = '/configs/agent_params.json'
DUMMY_ACTION_DISCRETE=0
DUMMY_ACTION_CONTINUOUS=[1.0, 1.0]
try:
    GYM_PORT=int(os.environ['GYM_PORT'])
except:
    GYM_PORT=8888


def action_space_type(config):
    if 'action_space_type' in config:
        if config['action_space_type'] not in ('discrete', 'continuous'):
            raise ValueError(
                f'Incorrectly defined action_space_type in config file.'
            )
        space_type = config['action_space_type']
    else:
        if isinstance(config['action_space'], list):
            # assuming discrete
            space_type = 'discrete'
        elif isinstance(config['action_space'], dict):
            # assuming continuous
            space_type = 'continuous'
        else:
            raise ValueError(
                f'Incorrectly defined action_space in config file.'
            )
    return space_type


class Server:
    def __init__(self, host='0.0.0.0', port=GYM_PORT):
        self.host = host
        self.port = port
        self.socket = zmq.Context.instance().socket(zmq.REP)
        self.socket.set(zmq.SNDTIMEO, 5000)
        self.socket.bind(f'tcp://{self.host}:{self.port}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.socket.close()

    def run(self):
        print('Starting server...')
        while True:
            packed_msg = self.socket.recv()
            msg = msgpack.unpackb(packed_msg)

            print('Received a message')
            print(msg)

            response = {'success': True}
            packed_response = msgpack.packb(response)
            self.socket.send(packed_response)


class GymAgent(ClippedPPOAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(AGENT_PARAMS_PATH, 'r') as file:
            config = json.load(file)

        assert 'action_space' in config, \
            f'Action space not defined in config file {AGENT_PARAMS_PATH}.'

        action_space = action_space_type(config)
        if action_space == 'discrete':
            self.dummy_action = DUMMY_ACTION_DISCRETE
        elif action_space == 'continuous':
            self.dummy_action = DUMMY_ACTION_CONTINUOUS
        else:
            raise ValueError(
                f'Action space can only be continuous or discrete. Got {action_space} instead.'
            )

        self.server = Server()
        self._previous_done = False
        self._hard_reset = False
        self._recieved_message = None
        print(f'================= Waiting for gym client =================')

        packed_msg = self.server.socket.recv()
        msg = msgpack.unpackb(packed_msg)
        print(f'=================== Gym Client Ready! ===================')

    def observe(self, env_response):
        response_dict = env_response.__dict__
        self._previous_done = response_dict['_game_over']
        if not self._hard_reset:
            packed_response = msgpack.packb(response_dict)
            self.server.socket.send(packed_response)

            packed_msg = self.server.socket.recv()
            self._recieved_message = msgpack.unpackb(packed_msg)
        else:
            self._recieved_message = {}
            if self._previous_done:
                self._hard_reset = False
                self._recieved_message['action'] = self.dummy_action  # IGNORED DUE TO RESET
            # else:
            #     env_response.game_over = True
            #     return True

    def act(self):
        if not self._hard_reset:

            if (self._recieved_message.get('action') is not None):
                action = self._recieved_message['action']

            elif (self._recieved_message.get('ready') is not None):
                self._hard_reset = True
                action = self.dummy_action

        else:
            action = self.dummy_action
        return ActionInfo(action=action)
