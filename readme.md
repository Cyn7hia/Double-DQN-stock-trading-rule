# Stock trading rule discovery with double deep Q-network
Code for paper Stock trading rule discovery with double deep Q-network
## Usage:

### Train Deep Q-learning:
```
python market_dqn.py ./kospi_10.csv dqn.h5 
```

### Test:
```
python market_model_test.py testlist.csv
```

env1: cash basis

env2: accrual basis
