# Matryoshka Representation Learning

An unofficial PyTorch implementation of [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) for contrastive self-supervised learning, specifically the [ReLIC](https://arxiv.org/abs/2010.07922) method. MRL encodes information at different granularities to learn flexible representations (single feature vector / embedding) of different dimensions that can be adapted to multiple downstream tasks. MRL can be easily used with other tasks and modalities such as classification, retrieval or language modeling. For example, ResNet50 returns a 2048 feature vector, where we can use the subset of that vector (eg. 64) for retrieval and a larger subset of the original vector (eg. 1024) for reranking. This can substantially reduce the computational resources.

The implementation is minimal and easily extendable with custom datasets. It shows that MRL blends very well with the ReLIC framework and is capable of learning very good representations. This repo doesn't depend on a specific self-supervised approach and can be easily extended to approaches as [BYOL](https://arxiv.org/abs/2006.07733) or [SimCLR](https://arxiv.org/abs/2002.05709).

# Results

Models are pretrained on training subsets - for `CIFAR10` 50,000 and for `STL10` 100,000 images. For evaluation, I trained and tested LogisticRegression on frozen features from:
1. `CIFAR10` - 50,000 train images on ReLIC
2. `STL10` - features were learned on 100k unlabeled images. LogReg was trained on 5k train images and evaluated on 8k test images.

Linear probing was used for evaluating on features extracted from encoders using the scikit LogisticRegression model. The table below shows training configurations and results when using the full dimension. Plots below show results accross dimensions.

More detailed evaluation steps and results for [CIFAR10](https://github.com/filipbasara0/matryoshka-representation-learning/blob/main/notebooks/linear-probing-cifar.ipynb) and [STL10](https://github.com/filipbasara0/matryoshka-representation-learning/blob/main/notebooks/linear-probing-stl.ipynb) can be found in the notebooks directory. 

| Evaulation model    | Dataset | Feature Extractor| Encoder   | Feature dim | Projection Head dim | Epochs | Top1 % |
|---------------------|---------|------------------|-----------|-------------|---------------------|--------|--------|
| LogisticRegression  | CIFAR10 | ReLIC            | ResNet-18 | 512         | 64                  | 400    | 84.19  |
| LogisticRegression  | STL10   | ReLIC            | ResNet-18 | 512         | 64                  | 400    | 81.55  |
| LogisticRegression  | STL10   | ReLIC            | ResNet-50 | 2048        | 64                  | 100    | 77.10  |

Below is the performance accross dimension for the ResNet18 model on the CIFAR10 dataset:

![image](https://github.com/filipbasara0/matryoshka-representation-learning/assets/29043871/0a25f5f4-b474-48e1-8314-6eafa90a942a)

Below is the performance accross dimension for the ResNet18 model on the STL10 dataset:

![image](https://github.com/filipbasara0/matryoshka-representation-learning/assets/29043871/e6aa56cc-00df-4bf6-b3c4-7dd7327a63db)

# Usage

### Instalation

```bash
$ pip install mrl-pytorch
```

Code currently supports ResNet18 and ResNet50. Supported datasets are STL10 and CIFAR10.

All training is done from scratch.

### Running Examples
`CIFAR10` ResNet-18 model was trained with this command:

`mrl_train --dataset_name "cifar10" --encoder_model_name resnet18 --fp16_precision`

`STL10` ResNet-50 model was trained with this command:

`mrl_train --dataset_name "stl10" --encoder_model_name resnet50 --fp16_precision`

### Detailed options
Once the code is setup, run the following command with optinos listed below:
`mrl_train [args...]⬇️`

```
ReLIC

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path where datasets will be saved
  --dataset_name {stl10,cifar10}
                        Dataset name
  -m {resnet18,resnet50}, --encoder_model_name {resnet18,resnet50}
                        model architecture: resnet18, resnet50 (default: resnet18)
  -save_model_dir SAVE_MODEL_DIR
                        Path where models
  --num_epochs NUM_EPOCHS
                        Number of epochs for training
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
  --fp16_precision      Whether to use 16-bit precision GPU training.
  --proj_out_dim PROJ_OUT_DIM
                        Projector MLP out dimension
  --log_every_n_steps LOG_EVERY_N_STEPS
                        Log every n steps
  --gamma GAMMA         Initial EMA coefficient
  --alpha ALPHA         Regularization loss factor
  --update_gamma_after_step UPDATE_GAMMA_AFTER_STEP
                        Update EMA gamma after this step
  --update_gamma_every_n_steps UPDATE_GAMMA_EVERY_N_STEPS
                        Update EMA gamma after this many steps
```

# Citation

```
@misc{kusupati2022matryoshka,
      title={Matryoshka Representation Learning}, 
      author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
      year={2022},
      eprint={2205.13147},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{mitrovic2020representation,
      title={Representation Learning via Invariant Causal Mechanisms}, 
      author={Jovana Mitrovic and Brian McWilliams and Jacob Walker and Lars Buesing and Charles Blundell},
      year={2020},
      eprint={2010.07922},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
