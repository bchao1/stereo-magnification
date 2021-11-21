FROM tensorflow/tensorflow:1.15.5-gpu

ADD . stereo-magnification
WORKDIR /stereo-magnification

RUN ls
RUN pip3 install -r requirements.txt
RUN pip3 list
RUN pwd

CMD [ "bash", "test_banana.sh" ]