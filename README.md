# BNSFRNPC
[Nana A. Otoo](https://github.com/naotoo1)

This repository contains the code for the paper Beyond Neural Scaling Laws For Fast Proven Robust Certification Of Nearest Prototype Classifiers.


The implementation requires Python >=3.11 . The author recommends to use a virtual environment or Docker image.
The details of the implementation and results evaluation can be found in the paper.

To install the Python requirements use the following command:
```python
pip install -r requirements.txt 
```

To replicate percetual-pruning and perceptual-training for cifar-10 run:
```python
python train.py --data_name cifar10 --model iglvq --train_norm lpips-l2 --test_norm l2  --feature_extraction --prune --prune_mode easy --prune_fraction 0.8 
python train.py --data_name cifar10 --model igtlvq --train_norm lpips-l2 --test_norm l2  --feature_extraction --prune --prune_mode hard --prune_fraction 0.2  
```

Users interested in replicating the results in the paper can run with the reported parematers in the paper by using:

```python
usage: train.py [-h] [--model MODEL] [--data_name DATA_NAME] [--test_size TEST_SIZE] [--train_norm TRAIN_NORM]
                [--test_norm TEST_NORM] [--proto_init PROTO_INIT] [--omega_init OMEGA_INIT] [--device DEVICE]
                [--ssl_metric SSL_METRIC] [--batch_size BATCH_SIZE] [--test_epsilon TEST_EPSILON]
                [--num_proto NUM_PROTO] [--prune_fraction PRUNE_FRACTION] [--prune_mode PRUNE_MODE]
                [--feature_extraction] [--prune] [--max_epochs MAX_EPOCHS] [--proto_lr PROTO_LR] [--omega_lr OMEGA_LR]
                [--noise NOISE]
```

To do post-training evaluation with pretained-models:
```python
usage: evaluate_script.py [-h] [--model MODEL] [--dataset DATASET] [--test_size TEST_SIZE] [--p_norm P_NORM]
                          [--metric METRIC] [--epsilon EPSILON] [--train_norm TRAIN_NORM]
```
