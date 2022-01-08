docker pull tensorflow/tensorflow:1.15.5-gpu
docker run -it -v /home/b08901039:/app -v /mnt/data/bchao/MPI:/mnt tensorflow/tensorflow:1.15.5-gpu /bin/bash