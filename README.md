docker build -t deeplearning-env .  

docker run --gpus all -p 4000:8000 -it deeplearning-env 