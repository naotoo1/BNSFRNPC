# BNSFRNPC
[Nana A. Otoo](https://github.com/naotoo1)

This repository contains the code for the paper Beyond Neural Scaling Laws For Fast Proven Robust Certification Of Nearest Prototype Classifiers.

## Abstract
Methods beyond neural scaling laws for beating power scaling laws in machine learning have become topical for high-performance machine learning models.
Nearest Prototype Classifiers (NPCs) introduce a category of machine learning models known for their interpretability. However, the performance of NPCs is frequently impacted by large datasets that scale to high dimensions. 
We surmount the performance hurdle by employing self-supervised prototype-based learning metrics to intelligently prune datasets of varying sizes, encompassing low and high dimensions. This process aims to enhance the robustification and certification of NPCs within the framework of the Learning Vector Quantization (LVQ) family of algorithms, utilizing Crammer normalization for arbitrary semi-norms (semi-metrics).
The numerical evaluation of outcomes reveals that NPCs trained with pruned datasets demonstrate sustained and enhanced performance compared to instances where training is conducted with full datasets. The self-supervised prototype-based metric (SSL) and the Perceptual-SSL (P-SSL) utilized in this study remain unaffected by the intricacies of optimal hyperparameter selection. Consequently, data pruning metrics can be seamlessly integrated with triplet loss training to assess the empirical and guaranteed robustness of\hspace{2pt} $L^{p}$-NPCs and Perceptual-NPCs (P-NPCs), facilitating the curation of datasets that contribute to research in applied machine learning.
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
