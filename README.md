# Beyond Neural Scaling Laws For Fast Proven Robust Certification Of Nearest Prototype Classifiers
<!-- BNSFRNPC -->
<!-- [Nana A. Otoo](https://github.com/naotoo1) -->

This repository contains the code for the paper "Beyond Neural Scaling Laws For Fast Proven Robust Certification Of Nearest Prototype Classifiers".


## ABSTRACT
_Methods beyond neural scaling laws for beating power scaling laws in machine learning have
become topical for high-performance machine learning models. Nearest Prototype Classifiers (NPCs)
introduce a category of machine learning models known for their interpretability. However, the
performance of NPCs is frequently impacted by large datasets that scale to high dimensions. We
surmount the performance hurdle by employing self-supervised prototype-based learning metrics to
intelligently prune datasets of varying sizes, encompassing low and high dimensions. This process
aims to enhance the robustification and certification of NPCs within the framework of the Learning
Vector Quantization (LVQ) family of algorithms, utilizing Crammer normalization for arbitrary
semi-norms (semi-metrics). The numerical evaluation of outcomes reveals that NPCs trained with
pruned datasets demonstrate sustained or enhanced performance compared to instances where training
is conducted with full datasets. The self-supervised prototype-based metric (SSL) and the Perceptual-
SSL (P-SSL) utilized in this study remain unaffected by the intricacies of optimal hyperparameter
selection. Consequently, data pruning metrics can be seamlessly integrated with triplet loss training
to assess the empirical and guaranteed robustness of Lp -NPCs and Perceptual-NPCs (P-NPCs),
facilitating the curation of datasets that contribute to research in applied machine learning_.


## Requirements

The implementation requires Python >=3.10 . We recommend to use a virtual environment or Docker image.
The details of the implementation and results evaluation can be found in the [paper](https://vixra.org/abs/2402.0027)
.

To install the Python requirements use the following command:
```python
pip install -r requirements.txt 
```
## Pruning, Robustification and Certification
To replicate SSL-pruning and training for mnist run:
```python
python train.py --dataset mnist --model iglvq --train_norm l2 --test_norm l2 --prune --prune_mode easy --prune_fraction 0.8 
python train.py --dataset mnist --model iglvq --train_norm l2 --test_norm l2 --prune --prune_mode hard --prune_fraction 0.2

python train.py --dataset mnist --model igtlvq --train_norm l2 --test_norm l2 --prune --prune_mode easy --prune_fraction 0.8
python train.py --dataset mnist --model igtlvq --train_norm l2 --test_norm l2 --prune --prune_mode hard --prune_fraction 0.2 
```
## Perceptual-Metric Pruning, Robustification and Certification
To replicate percetual SSL-pruning and perceptual-training for cifar-10 run:
```python
python train.py --dataset cifar10 --model iglvq --train_norm lpips-l2 --test_norm l2  --feature_extraction --prune --prune_mode easy --prune_fraction 0.8 
python train.py --dataset cifar10 --model iglvq --train_norm lpips-l2 --test_norm l2  --feature_extraction --prune --prune_mode hard --prune_fraction 0.2  
```

Users interested in replicating the results in the paper can run with the reported parematers in the paper by using:

```python
usage: train.py [-h] [--model MODEL] [--data_name DATASET] [--test_size TEST_SIZE] [--train_norm TRAIN_NORM]
                [--test_norm TEST_NORM] [--proto_init PROTO_INIT] [--omega_init OMEGA_INIT] [--device DEVICE]
                [--ssl_metric SSL_METRIC] [--batch_size BATCH_SIZE] [--test_epsilon TEST_EPSILON]
                [--num_proto NUM_PROTO] [--prune_fraction PRUNE_FRACTION] [--prune_mode PRUNE_MODE]
                [--feature_extraction] [--prune] [--max_epochs MAX_EPOCHS] [--proto_lr PROTO_LR] [--omega_lr OMEGA_LR]
                [--noise NOISE]
```
## Robustness Evaluation
The evaluate_script.py generates reports on Clean Test Error (CTE), Lower Bound on the Robust Test Error (LRTE), and Upper Bonund on the Robust Test Error (URTE) 
for a specified NPC. It is tailored for utilization on either newly trained models or pre-trained models. It features multiple parameters, with default values contingent upon other parameters. 
To illustrate an example of post-training robustness evaluation with a pre-trained, run:
```python
python evaluate_script.py --model iglvq --dataset cifar10 --train_norm lpips-l2 --epsilon 0.1412 --test_size 0.2 --p_norm l2
```
