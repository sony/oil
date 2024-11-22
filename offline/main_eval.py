
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import torch
import pickle
from definitions import ROOT_DIR



if __name__ == "__main__":
    
    algo = "bc"
    exp_path = ROOT_DIR / "offline" / "bc_seed_0" / "model_final" / "bc_model.pth"
    normalize_path = ROOT_DIR / "offline" / "bc_seed_0" / "normalize_dict.pkl"
    

    model = torch.jit.load(exp_path)
    with open(normalize_path, "rb") as f:
        normalize_dict = pickle.load(f)
        
    