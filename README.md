# HybridPipe

Given an human-generated pipeline (HI-pipeline) for an ML task, HybridPipe introduces a reinforcement learning based approach to search an optimized  ML-generated pipeline (ML-pipeline) and adopts an enumeration-pruning strategy to carefully select the best performing combined pipeline.

## Requirements

This code is written in Python. Requirements include

- python = 3.8.12
- numpy = 1.19.5 
- pandas = 1.1.3
- torch = 1.9.0
- scikit-learn = 0.23.2

You can install multiple packages:

```
pip install -r requirements.txt
```



## Quick Start

### run example to generate hybrid-pipeline

```
python example.py
```

The file `example.py` is an example. Modify it according to your configuration.

- The **input** contains 1 notebook, 1 csv file, some information about the ML task (including model and label_index)
- The **output** is the best hybrid program,the accuracy of HI-Pipe and the accuracy of Hybrid-Pipe.

#### Customized input

You can put new data in the folder `$hybridpipe_HOME/data/`and organize them as:

```python
hybridpipe
├── data
│   ├── dataset 
│   │   └── new_folder   #input dataset(csv file)
│   └── notebook #input notebook
├── MLPipeGen
└── HybridPipeGen
```

Then modify the input in example.py.



## Experimental dataset

All experimental datasets can be downloaded via https://www.dropbox.com/s/6xy6m70hjm3bz2l/exp_dataset.zip?dl=0 .

And the experimental datasets is organized as:

```python
exp_dataset
├── KGTorrent #input file
│   ├── dataset #all csv file
│   └── notebook #all Jupyter notebooks
├── hipipe_code #.py file of hipipe
├── end_index # the last position of hipipe that can be inserted into the ML operation
├── hipipe_graph #the graph of hipipe by profiling hipipe_code
├── hipipe_result #the result of hipepie on test dataset
├── candidate_hybridpipe_test_code #the code of candidate hybridpipe on test dataset
├── candidate_hybridpipe_validation_code #the code of candidate hybridpipe on validation dataset
├── candidate_hybridpipe_validation_result #the result of candidate hybridpipe on validation dataset
└── best_hybridpipe_test_result #the result of selected best hybridpipe on test dataset
```

KGTorrent is the input data, which includes raw dataset and notebook, and other intermediate data can be obtained by our program. Considering that it takes a lot of time to reproduce our experiments from scratch, we provide intermediate data. 

The final evaluation results of hybrid_pipeline are stored in the `best_hybridpipe_test_result` folder, and each folder under this folder is named notebook, which stores the accracy score of each notebook's hybrid_pipeline. 

