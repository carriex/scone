# Sequence-to-sequence model for ALCHEMY dataset 

This repo contains a sequence to sequence model to map instructions to action in the [ALCHEMY environment](https://nlp.stanford.edu/projects/scone/).

More details on the model can be found in the [blog entry](https://carriex.github.io/me/blog/entry1.html).

### Prerequisite

1. Clone the repo 

```bash
$ git clone https://github.com/carriex/scone.git
```

2. Install the python dependency packages 
```bash
$ pip install -r requirements.txt 
```

### Train the sequence to sequence model


```bash
usage: train_pl.py [-h] [--train_data TRAIN_DATA] [--val_data VAL_DATA]
                   [--test_data TEST_DATA]
                   [--test_interaction_label TEST_INTERACTION_LABEL]
                   [--test_instruction_label TEST_INSTRUCTION_LABEL]
                   [--test_result TEST_RESULT]
                   [--train_batch_size TRAIN_BATCH_SIZE]
                   [--test_batch_size TEST_BATCH_SIZE]
                   [--val_batch_size VAL_BATCH_SIZE] [--num_worker NUM_WORKER]
                   [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS]
                   [--teacher_forcing_ratio TEACHER_FORCING_RATIO]
                   [--num_epoches NUM_EPOCHES] [--lr LR]
                   [--load_from LOAD_FROM] [--run_test] [--unit_test]
                   [--use_attention] [--model_name MODEL_NAME]
                   [--version_name VERSION_NAME] [-p PERCENT_CHECK]

optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
  --val_data VAL_DATA
  --test_data TEST_DATA
  --test_interaction_label TEST_INTERACTION_LABEL
  --test_instruction_label TEST_INSTRUCTION_LABEL
  --test_result TEST_RESULT
  --train_batch_size TRAIN_BATCH_SIZE
                        training batch size
  --test_batch_size TEST_BATCH_SIZE
                        testing batch size
  --val_batch_size VAL_BATCH_SIZE
                        validation batch size
  --num_worker NUM_WORKER
                        number of worker for dataloader
  --hidden_size HIDDEN_SIZE
                        size of the hidden layer of RNN
  --num_layers NUM_LAYERS
                        number of layers of the RNN
  --teacher_forcing_ratio TEACHER_FORCING_RATIO
                        teacher forcing ratio
  --num_epoches NUM_EPOCHES
                        number of training epoches
  --lr LR               initial learning rate
  --load_from LOAD_FROM
                        path to pretrained model checkpoint
  --run_test            test on pretrained model
  --unit_test           run fast dev
  --use_attention       use attention for decoder
  --model_name MODEL_NAME
                        model name
  --version_name VERSION_NAME
                        version name
  -p PERCENT_CHECK, --percent_check PERCENT_CHECK
                        use only a certain percentage of data to run
```

### Evaluate the result

Output the accuracy of the predicted actions. By default evaluation on instruction level and interaction level will be run at the end of the training to the ```results``` directory.

#### Output result and evaluate on trained model

```bash
$ python train_pl.py --load_from <path_to_checkpoint> --run_test
```

#### Evaluate on output csv file
```bash
# Instruction level  
$ python evaluate.py -p results/test_pred_instruction.csv -l data/test_instruction_y.csv

# Interaction level
$ python evaluate.py -p results/test_pred_interaction.csv -l data/test_interaction_y.csv
```


