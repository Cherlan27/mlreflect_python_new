FROM  nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
WORKDIR /root/my_dir
COPY requirements.txt ./

ADD . /mlreflect
WORKDIR /mlreflect

RUN apt-get update && apt-get upgrade -y &&\
	apt-get install -y python3 python3-pip
	
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
RUN pip3 install -e mlreflect
RUN pip3 install -e refnx
RUN pip3 install symfit

ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA-DRIVER-CAPABILITIES compute,utilitydocke