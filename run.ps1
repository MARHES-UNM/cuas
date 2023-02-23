set-variable -name DISPLAY -value 192.168.1.29:0.0
# docker run --rm -it -v ${pwd}:/workspace/cuas -p5900:5900 -e DISPLAY=$DISPLAY --gpus all  rayproject/ray-ml:latest-gpu
docker run --rm -it -v ${pwd}:/home/dev/workspace/cuas -p5900:5900 -e DISPLAY=$DISPLAY --gpus all  cuas:1.0 bash