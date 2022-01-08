FROM tensorflow/tensorflow:1.15.5-gpu

ADD . stereo-magnification
WORKDIR /stereo-magnification

RUN pip3 install -r requirements.txt
RUN pip3 list

CMD [ "bash", "train.sh" ]