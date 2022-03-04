# Pytorch_template

pytorch basic template for AI



# Description

This repository provides simple template for AI models.

You can see the structure [here](#structure)

You can see the way to use [here](#how-to-use)

You can see the outputs [here](#outputs)

# Structure

```bash
├──── baseline
	├──── dataset
	│	└──────── dataset.py
	│
	├──── models
	│	├──────── loss.py
	│	└──────── model.py
	│
	├──── trainer
	│	└──────── trainer.py
	│
	└──── utils
		├──────── activate.py
		├──────── gradcam.py
		├──────── mainFunctions.py
		└──────── util.py

```

# How to use

- **Moderate config.cfg**

  You first need to moderate config.cfg file in the baseline directory.

  there's 5 sections in config file and each section includes settings for AI models.

  - **path section**

    `current_path` : base path of the baseline directory

    `save_dir` : pth files are saved in 'current_path/save_dir'

    `train_df_path` : csv file path which includes information about training images

    `val_df_path `: csv file path which includes information about validation images

    `test_df_path` : csv file path which includes information about test images

  - **net section**

    `model` : model name you want to use (this model should be in timm library)

    `criterion` : loss function you want to use (if you want to use customized function, you should include the function in models/loss.py file)

    `metric` : its not been updated yet...

    `optimizer` : optimizer you want to use

    `num_classes` : you can customize the model by changing this parameter

    `lr` :  learning rate

    `epoch` : number of epochs you want to train

    `lr_scheduler` : its not been updated yet... (default : None)

    `batch_size` : batch size of your training/validation/test datas

    `freeze` : you can set this to True if you want to freeze pretrained model

    `pretrained` : you can set this to True if you want to use pretrained model

    `img_shape` : it must consists of two numbers with comma in the center (w, h)

  - **train section**

    `early_stopping` : early stopping limit

  - **mode section**

    `mode` : train for training, test for test

  - **project section**

    `name` : wandb project name

    `experiment_key` : key words for wandb experiment name (it must be seperated by comma and this automatically combines keywords for experiment name)

- **Run**

  ```
  python main.py --config config.cfg
  ```



# Outputs

- **Log**

  This directory can be checked after running the code. Log files will be saved here.

- **experiment**

  This directory can be checked after running the code. pt files will be saved here.

  (pth files are saved every epochs and best_pth file will be included as {model_name}_best_model.pt)

