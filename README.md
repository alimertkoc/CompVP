# CompVP
FAU - Computational Visual Perception Project

# Authors 
- Ali Mert Koc (ok81isiz)
- Rasul Naghizade (sy18qubo)
- Ozan Sahin (ek21inoj)

# Important Note
Project should be run after navigating into ```/code``` directory. Because it will look for the dataset folder ```/data``` under ```/code``` directory. If it will not find any, it will start to download automatically. If you don't run the code under ```/code```, it will try to download dataset under ```/Part1```. 
1. ```cd Part1/code```
2. ```python main.py```

# Project notes
Dataloaders can be found in ```main.py```
Low initial visual acuity property is implemented in ```VisualAcuityDataset.py```
Limited color perception property is implemented in ```LimitedColorPerceptionDataset.py```
The dataloader evaluated in ```test_performance.py```
