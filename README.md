# TSF Skeleton

This project provides a highly customizable time-series forecasting (TSF) pipeline. It also includes a large collection of forecasting baseline models ranging from basic MLP, RNN to novel Transformer-based models and DLinear. 

## Why?

During our research on the Time-series forecasting topic, we find that the open-source repositories often adopts totally different pipeline, which takes a lot of time to read and debug just in order to run a baseline experiment on it. Recently, we find that a series of work prefer to develop on our repository [LSTF-Linear](https://github.com/cure-lab/LTSF-Linear). Thus, we make the TSF Skeleton based on it as a baseline repository and a easy to use pipeline. 



## Features
* **Baseline Results:** We validate every models in this repository to make sure the [results](#results) is similiar to the original implementation. You can also directly run these model with costomized experiment settings. 
* **Simple Pipeline: ** We keep the pipeline as concise as possible. So that you can make modification and realize your idea without too much learning cost. 
* **Forecasting Models:** A large collection of forecasting models; from transformer models (such as 
 Informer) to deep learning models (such as RNN). See [table of models below](#forecasting-models).
* **Multivariate Support:** Datasets contain multiple time-varying dimensions instead of a single scalar value. Many models can consume and produce multivariate series.
* **Regression Models:** It is possible to plug-in any scikit-learn compatible model
  to obtain forecasts as functions of lagged values of the target series and covariates.
* **Data processing:**  This repository has its own dataloader to preprocess the datasets. 
* **Metrics:** A variety of metrics for evaluating time series' goodness of fit;
  from RMSE to Mean Absolute Scaled Error.
* **Datasets** In the dataset folder, we offer some popular time series datasets for rapid
  experimentation (such as ETT).
* **cfgs sysyem**  In this repository, we create a unique configuration file system. we have provided paremeters of all available datasets, models and expression files. All parameters initially are default. You can also change it according to your needs.
* **Documentation (Working on): ** We provide detailed documentation about not only how to use, but also how to develop based on TSF Skeleton. 

## Forecasting Models

Here's a breakdown of the forecasting models in our current repository . We are constantly working
on bringing more models and features.

Model | Univariate | Multivariate |UseTimeFeature| Reference|
--- | --- | --- | --- | ---| 
`RNN` | ✅ | ✅ |  | [DeepAR paper](https://arxiv.org/abs/1704.04110)
`SCINet` | ✅ | ✅ |  |[SCINet paper](https://arxiv.org/pdf/2106.09305.pdf)|
`MTGNN` | ✅ | ✅ | ✅ | [MTGNN paper](https://arxiv.org/abs/2005.11650)
`InceptionTime` | ✅ | ✅ | | [InceptionTime paper](https://arxiv.org/pdf/1909.04939.pdf)
`STGCN`  | ✅ | ✅ | ✅ |[STGCN paper](https://www.ijcai.org/proceedings/2018/0505)
`FNN` | ✅ | ✅ | ✅ |[FNN paper](https://arxiv.org/abs/2002.05909)|
`DCRNN` | ✅ | ✅ | ✅ |[DCRNN paper](https://arxiv.org/abs/1707.01926)|
`Autoformer` | ✅ | ✅ | ✅ |[Autoformer paper](https://arxiv.org/abs/2106.13008)|
`MLP` | ✅ | ✅ |  ||
`gMLP` | ✅ | ✅ |  ||
`ResNet` | ✅ | ✅ |  |[ResNet paper](https://arxiv.org/abs/1512.03385)|
`TCN` | ✅ | ✅ |  |[TCN paper](https://github.com/locuslab/TCN)|

## Dataset
We conduct the experiments on **7** popular time-series datasets, namely **Electricity Transformer Temperature (ETTh1, ETTh2 , ETTm1 and ETTm2) , and Solar-Energy, Electricity and Exchange Rate**, from **power, energy, finance and traffic domains**. 

### Overall information of the 7 datasets

| Datasets      | Variants | Timesteps | Granularity | Start time | Task Type   |
| ------------- | -------- | --------- | ----------- | ---------- | ----------- |
| ETTh1         | 7        | 17,420    | 1hour       | 7/1/2016   | Multi-step  |
| ETTh2         | 7        | 17,420    | 1hour       | 7/1/2016   | Multi-step  |
| ETTm1         | 7        | 69,680    | 15min       | 7/1/2016   | Multi-step  |
| ETTm2         | 7        | 69,680    | 15min       | 7/1/2016   | Multi-step  |
| Solar-Energy  | 137      | 52,560    | 1hour       | 1/1/2006   | Single-step |
| Electricity   | 321      | 26,304    | 1hour       | 1/1/2012   | Single-step |
| Exchange-Rate | 8        | 7,588     | 1hour       | 1/1/1990   | Single-step |


### Parameter highlights

| Parameter Name | Description                  | Default                    |
| -------------- | ------------------ | -------------------------- |
| path      | The root path of datasets  | N/A |
| dataset_name | dataset                 | N/A |
| horizon       | Horizon                      |24 |
| lookback        | Look-back window           | 48 |
| channel             | features of dataset                |  7   |
| train_ratio    | part of dataset used for trainning | 0.6|
| valid_ratio    | part of dataset used for validation| 0.2|
| test_ratio     | part of dataset used for test      | 0.2|
| multivariate   | support multivariate      | true|
| freq           | Freq for time features encoding (defaults to h). This can be set to s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h| h|
|target          | Target feature in S or MS task (defaults to OT)| OT|
|scalar          | scalar of the dataset| StandardScalar|
|normalize       | Normalize            | 2|
 



## Get started

### Requirements

Install the required package first:

```
cd REPO_skeleton
conda create -n REPO_skeleton python=3.8
conda activate REPO_skeleton
pip install -r requirements.txt
```

### Dataset preparation
All datasets can be downloaded [here](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing). 
[![ett](https://img.shields.io/badge/Download-ETT_Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/1NU85EuopJNkptFroPtQVXMZE70zaBznZ)
[![financial](https://img.shields.io/badge/Download-financial_Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/12ffxwxVAGM_MQiYpIk9aBLQrb2xQupT-)

The data directory structure is shown as follows. 
```
./
└── datasets/
    ├── ETT-data
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   ├── ETTm1.csv
    |   └── ETTm2.csv
    ├── financial
    │   ├── electricity.txt
    │   ├── exchange_rate.txt
    │   ├── solar_AL.txt
    
```

### Quick start

This project provides highly customizable time-series model training backbone. If you want to configure your model manually, please refer to our example json config files under the path `cfgs/`, which will give you some insight about how to write a json config file. To use a custom dataset, you can put the dataset file under the `datasets/` directory for better file management, then don't forget to specify the path in the config file. Remember to also install the dependencies, follow the steps [here](https://github.com/VEWOXIC/REPO_skeleton##requirements). After that, you just have to tell the program where to find your config file. Here is an example:

```
python run.py --cfg_file 'path_to_your_json_file.json'
```

However, if you want to try out with our provided examples, you can follow the steps below:

1. Download our example dataset file from [here](https://github.com/VEWOXIC/REPO_skeleton#dataset-preparation). 
    * In this demo, we use the `ETTh1.csv` dataset file under the path `dataset/ETT-data/ETT/` in our provided link.
2. Put the downloaded dataset file under `datasets/` directory.
3. Follow the steps in the [requirements](https://github.com/VEWOXIC/REPO_skeleton##requirements) section to install the dependencies.
4. Run the program with the following commands: 
```
python run.py --cfg_file 'cfgs/exp/SCINet/SCINet_ETTh1_mult_s48h24.json'
```

The above demo trains with a SCINet model and a Electricity Transformer Temperature dataset. 
  * Details can be found in the config file `cfgs/exp/SCINet/SCINet_ETTh1_mult_s48h24.json`.


## Results

We have updated the experiment results of all models on 7 datasets. 

||| SCINET | RNN | InceptionTime |TransformerModel | STGCN  | FNN | AutoEncoder | MLP  | gMLP | FCN | ResNet | TCN | OmniScaleCNN | 
|-------------|---------|------------|-----------|-----------|-----------|-----------|----------|----------|----------|----------|----------|----------|----------|----------|
|Dataset|Metric|
|ETTh1 | MAE|0.387|0.390|0.433|0.413|0.470|0.374|0.498|0.530|0.405|0.490|0.459|0.416|0.398|
| | MSE|0.358|0.353|0.401|0.383|0.500|0.345|0.520|0.568|0.379|0.497|0.451|0.398|0.363|
| |RSE| 0.534|0.530|0.564|0.552|0.631|0.524|0.643|0.672|0.615|0.629|0.599|0.563|0.537|
| |CORR| 0.796|0.797|0.774|0.775|0.701|0.801|0.693|0.654|0.779|0.707|0.733|0.753|0.789|
|ETTh2 |MAE|0.279|0.271|0.278|0.269|0.313|0.252|0.349|0.419|0.260|0.338|0.303|0.269|0.267|
| | MSE|0.177|0.165|0.169|0.163|0.228|0.150|0.251|0.388|0.158|0.246|0.206|0.166|0.161|
| |RSE|0.446|0.430|0.436|0.428|0.505|0.411|0.531|0.660|0.422|0.526|0.480|0.432|0.426|
| |CORR|0.738|0.737|0.730|0.739|0.666|0.750|0.665|0.610|0.757|0.650|0.672|0.744|0.744|
|ETTm1 | MAE|0.223 |0.238|0.217|0.232|0.256|0.216|0.255|0.304|0.209|0.232|0.219|0.217|0.219|
| |MSE|0.119|0.124|0.113|0.124|0.150|0.114|0.134|0.184|0.106|0.117|0.108|0.116|0.118|
| |RSE|0.363|0.372|0.354|0.371|0.409|0.351|0.385|0.452|0.343|0.360|0.346|0.146|0.362|
| |CORR|0.828|0.804|0.825|0.803|0.779|0.815|0.798|0.740|0.835|0.805|0.829|0.822|0.809|
|ETTm2 |MAE|0.397|0.408|0.386|0.418|0.382|0.378|0.367|0.448|0.378|0.402|0.390|0.214|0.402|
||MSE|0.412|0.448|0.400|0.455|0.326|0.383|0.337|0.479|0.391|0.410|0.405|0.118|0.436|
| |RSE|0.572|0.596|0.564|0.601|0.574|0.551|0.517|0.617|0.557|0.571|0.567|0.352|0.589|
| |CORR|0.788|0.771|0.797|0.755|0.774|0.791|0.825|0.765|0.805|0.786|0.792|0.825|0.769|
