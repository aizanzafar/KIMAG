# KI-MAG: A Knowledge-Infused Abstractive Question Answering System in Medical Domain

## Content
1. [Requirement](#requirement)
2. [Training and Finetuning](#training-finetuning)
3. [Evaluation](#evaluation)


## Requirement

Installing PyTorch:

```
pip install torch
```

Setting up Transfomers:

Run the following command to clone the transformer repository, change to the desired version and install it.

```
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 7b75aa9fa55bee577e2c7403301ed31103125a35
pip install -e .
```

Replace modeling_bart.py and modeling_utils.py inside src/transformers (of the library installed in the previous step) with the scripts in the bart directory.

## Training and Finetuning

```
python cli.py --do_train --output_dir out/${data}_checkpoint \
        --checkpoint ${checkpoint_path} \
        --train_file data/${data}/train.json \
        --predict_file data/${data}/dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --append_another_bos --do_lowercase
```

Use --checkpoint to specify the path to the checkpoint (if no path is passed, bart-large is loaded by default)
Use --train_file to specify the path to the training data.
Use -- predict_file to specify the path to the dev data.
Use --train_batch_size and --predict_batch_size to specify the training and prediction batch sizes.
The script will save the best checkpoint inside `out/${data}_checkpoint`.
Check cli.py for other command line arguments.

## Evaluation

Run the following command for evaluation

```
python cli.py --do_predict --output_dir out/${data}_checkpoint \
        --predict_file data/${data}/test.json \
        --predict_batch_size ${test_bs} \
        --append_another_bos --prefix test_
```

embeddings.json file uploaded at: https://drive.google.com/file/d/1KWiLaLPvEfRSPNQ0JDMA4Rx6uwtWaKsi/view?usp=sharing

Use --predict_file to specify the path to the file to be used for prediction.
The above command uses the checkpoint out/${data}_checkpoint/best-model.pt .
The predictions are saved inside out/${data}_checkpoint .





