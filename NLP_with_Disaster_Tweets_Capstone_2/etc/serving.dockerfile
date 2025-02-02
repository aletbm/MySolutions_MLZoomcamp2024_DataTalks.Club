FROM tensorflow/serving:2.14.0

ENV MODEL_NAME=disaster_tweets_model
COPY ../scripts/disaster_tweets_model ./models/disaster_tweets_model/1

EXPOSE 8500
EXPOSE 8501
#To build it
#docker build -t tf-serving-disaster-tweets-model -f ./etc/serving.dockerfile .

#To do tag the image docker
#docker tag tf-serving-disaster-tweets-model aletbm/tf-serving-disaster-tweets-model

#To do push it
#docker push aletbm/tf-serving-disaster-tweets-model:latest
