# LocaleGN

This is the original pytorch implementation of LocaleGN in the following paper: 




<p align="center">
  <img width="350" height="400" src=LocaleGN_diagram.png>
</p>

## Requirements
- python 3
- see `requirements.txt`


## Data Preparation

### Step1: Download METR-LA, PEMS-BAY, PEMS04, PEMS07 and PEMS08 data from links provided by [StemGNN](https://github.com/microsoft/StemGNN). Download ST data from link provided by https://github.com/zhiyongc/Seattle-Loop-Data           

### Step2: Process raw data 

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
## Train Commands

```
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
```


