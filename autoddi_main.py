from autoddi.auto_model import AutoModel
from set_config import data_name, save_suffix, search_parameter, gnn_parameter
from planetoid import Planetoid
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

graph = Planetoid(data_name, save_suffix=save_suffix)

AutoModel(graph, search_parameter, gnn_parameter, save_suffix)
