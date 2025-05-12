#!/bin/bash
mkdir data

curl -L -o ./data/fer-2013-facial-expression-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/pankaj4321/fer-2013-facial-expression-dataset

unzip data/fer-2013-facial-expression-dataset.zip -d data
  
