FROM trial_plimac3

RUN conda install -c anaconda setuptools
RUN pip install --upgrade pip && \
    pip install tensorflow==1.14.0 --user && \
    pip install ml_metrics==0.1.4 && \
    pip install --upgrade scipy==1.1.0 && \
    conda clean --all && \
    conda install pytorch==1.0.0 torchvision==0.2.1 cpuonly -c pytorch && \
    pip install torchtext==0.4.0 && \
    pip install attrdict==2.0.1 && \
    pip uninstall --yes numpy && \
    pip install numpy==1.16.4 && \
    pip uninstall --yes gast && \
    pip install gast==0.2.2

WORKDIR /workdir
RUN git clone https://github.com/mack-the-psych/vdok3.git

RUN echo "/workdir/vdok3/prep" > /opt/conda/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/workdir/vdok3/extract" >> /opt/conda/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/workdir/vdok3/process" >> /opt/conda/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/workdir/vdok3/reorganize" >> /opt/conda/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/workdir/vdok3/train" >> /opt/conda/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/workdir/vdok3/train/pytorch_advanced/nlp_sentiment_bert" >> /opt/conda/lib/python3.6/site-packages/vdok3-custom.pth

WORKDIR /workdir/vdok3/train/pytorch_advanced/nlp_sentiment_bert
RUN python make_folders_and_data_downloads.py
WORKDIR /workdir
