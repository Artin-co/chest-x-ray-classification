

FROM gpuci/miniconda-cuda:11.1-runtime-centos7

ADD code.yaml /home/bin/code.yaml
RUN conda env create -f /home/bin/code.yaml

RUN mkdir -p /home/project/
WORKDIR /home/project


CMD [ "bash" ]
