FROM tensorflow/serving:2.14.0

ENV MODEL_NAME=blood-cell-model
COPY ../scripts/blood-cell-model /models/blood-cell-model/1

#To build it
#docker build -t tf-serving-blood-cell-model -f ./etc/serving.dockerfile .

#To do tag the image docker
#docker tag blood_cell_cancer_prediction aletbm/tf-serving-blood-cell-model

#To do push it
#docker push aletbm/tf-serving-blood-cell-model:latest