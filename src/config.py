import yaml
import munch

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_path = './config/model.yaml'
yaml_config = load_config_from_yaml(config_path)
service_yaml_config = load_config_from_yaml('./config/service.yaml')

class SelfPlayConfig:
    def __init__(self):
        self.game_num = 100
        #self.mcts_sims = 800
        self.mcts_sims = 1000
        self.max_depth = 200
        self.num_clients = 50
        self.noise_eps = 0.5
        self.dir_alpha = 0.1
        self.wgt_p = 1.0
        self.virtual_loss = 3.0
        # make sure win_reward is greater than other reward/loss
        self.win_reward = 5.0
        #self.resigned_threshold = 0.8
        self.min_resigned_turn = 5
        self.tau_decay_rate = 0.99
        self.resign_threshold = -0.8
        self.max_game_length = 500

class ModelConfig:
    def __init__(self, yaml_config):
        self.cnn_filter_num = yaml_config.get('cnn_filter_num', 256)
        self.cnn_first_filter_size = yaml_config.get('cnn_first_filter_size', 5)
        self.cnn_filter_size = yaml_config.get('cnn_filter_size', 3)
        self.res_layer_num = yaml_config.get('res_layer_num', 5)#
        self.l2_reg = yaml_config.get('l2_reg', 1e-4)
        self.learning_rate = yaml_config.get('learing_rate', 1e-4)
        self.batch_size = yaml_config.get('batch_size', 128)
        self.epochs = yaml_config.get('epochs', 10)
        self.log_interval = yaml_config.get('log_interval', 100)
        self.value_fc_size = yaml_config.get('value_fc_size', 256)
        self.distributed = yaml_config.get('distributed', False)
        self.input_depth = yaml_config.get('input_depth', 18)
        self.board_width = yaml_config.get('board_width', 9)
        self.board_height = yaml_config.get('board_height', 10)

class ServiceConfig:
    def __init__(self, yaml_config):
        self.batch_size_serve = yaml_config.get('batch_size_serve', 256)
        self.batch_timeout = yaml_config.get('batch_timeout', 0.3)
        
config = munch.munchify(
    {
        'model': ModelConfig(yaml_config),
        'service': ServiceConfig(service_yaml_config),
        'self_play': SelfPlayConfig()
    }
)