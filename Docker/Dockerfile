FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

WORKDIR /scratch/IDF

RUN apt-get update && apt-get install -y \
    wget git bzip2 vim && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:$PATH
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

RUN conda config --set ssl_verify no && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Disable pip SSL verification
RUN pip config set global.trusted-host "pypi.org" && \
    pip config set global.trusted-host "files.pythonhosted.org" && \
    pip config set global.disable-pip-version-check true && \
    pip config set global.timeout 120

# Create conda environment
COPY requirements.txt /scratch/IDF/
RUN conda create -n idf python=3.10 -y

# Install requirements inside idf
RUN conda run -n idf pip install -r /scratch/IDF/requirements.txt

# Copy project code
COPY . /scratch/IDF


CMD ["bash"]
