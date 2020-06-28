FROM trial_plimac3

RUN conda install -c anaconda setuptools
RUN pip install --upgrade pip && \
    pip install tensorflow==1.14.0 --user && \
    pip install ml_metrics==0.1.4 && \
    pip install --upgrade scipy==1.1.0

WORKDIR /workdir
RUN git clone https://github.com/mack-the-psych/vdok3.git
