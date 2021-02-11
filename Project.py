from dataclasses import dataclass
from pathlib import Path

@dataclass
class Project:
    """
    This class represents our project. It stores useful information about the structure, e.g. paths.
    """
    base_dir: Path = Path(__file__).parents[0]
    data_dir = base_dir / 'dataset'
    checkpoint_dir = base_dir / 'checkpoint'
    crossValidFolds = 3
    inpCSV = data_dir / 'angles_UP_filtered_v2.csv'
    inpDictDir = base_dir / 'dataset'
    outDictDir = base_dir / 'dataset'
    maxIte = 4
    productName = 'N34V'
    inpDictName = productName + '_InputDict.pkl'
    setGPU = 'cuda:1'
    specificTargets = ['Associated_angle_model']
    testSetStart = '2020-07-25 00:00:00'
    timeKey = 'Time'
    valSetOption = 2
    networkParams = {
        "lr" : 0.00005,
        "n_hidd" : 32,
        "n_hidd_layers" : 1,
        "act_func" : 'relu',
        "batch_size" : 16,
        "momentum" : 0.9,
        "weight_decay" : 0.0,
        "l1_reg" : 0.0001,
        "num_epc" : 5,
        "dropout" : False, 
    }

    def __post_init__(self):
        # create the directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
