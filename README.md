# HybridPipe

Given an human-generated pipeline (HI-pipeline) for an ML task, HybridPipe introduces a reinforcement learning based approach to search an optimized  ML-generated pipeline (ML-pipeline) and adopts an enumeration-pruning strategy to carefully select the best performing combined pipeline.

## Requirements

This code is written in Python. Requirements include

- python = 3.8.12
- numpy = 1.19.5 
- pandas = 1.1.3
- torch = 1.9.0
- scikit-learn = 0.23.2

You can install multiple packages in conda:

```
conda install --yes --file requirements.txt
```



## Quick Start

### run example to generate hybrid-pipeline

```
python example.py
```

The file `example.py` is an example. Modify it according to your configuration.

- The **input** contains 1 notebook, 1 csv file, some information about the ML task (including model and label_index)
- The **output** is  the accuracy of HI-Pipe and the accuracy of Hybrid-Pipe, the hybrid program, and the data prep pipeline of the input notebook(Optional).
  The data prep pipeline will be saved at `HybridPipeGen/core/profiling_result/example.png`.

#### Customized input

You can put new data in the folder `$hybridpipe_HOME/data/`and organize them as:

```
hybridpipe
├── data
│   ├── dataset 
│   │   └── new_folder   #input dataset(csv file)
│   └── notebook #input notebook
├── MLPipeGen
└── HybridPipeGen
```

Then modify the input in example.py.

