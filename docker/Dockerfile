# # Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04
LABEL author="privisaa@amazon.com"

ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"


RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Reinstall torch
RUN pip install --upgrade --force-reinstall  torch==1.8.1 torchvision==0.9.1 cython boto3

# Install MMCV, MMDetection and MMSegmentation
RUN export MKL_SERVICE_FORCE_INTEL=1 && pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.1/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# Install MMDetection3D
RUN conda clean --all
RUN git clone https://github.com/iprivit/mmdetection3d.git /mmdetection3d
WORKDIR /mmdetection3d
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN mkdir /opt/ml/ && mkdir /opt/ml/code

COPY train.py /opt/ml/code/train.py

RUN wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/3dssd/3dssd_kitti-3d-car_20210602_124438-b4276f56.pth -O /opt/ml/code/3dssd_kitti-3d-car_20210602_124438-b4276f56.pth

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM /opt/ml/code/train.py

RUN export MASTER_ADDR=algo-1
RUN export MASTER_PORT=12345

# Starts PyTorch distributed framework
ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]