# assert args.fusion_model in ['SAN', 'MLP', 'BAN', 'UD', 'LSTMEncoder']

from .fusion_updn import UD
from .fusion_ban import BAN
from .fusion_san import SAN
from .fusion_mlp import MLP
from .relation_lstm import LSTMEncoder