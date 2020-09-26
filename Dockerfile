FROM trial_plimac3

RUN conda install -c anaconda setuptools
RUN pip install --upgrade pip && \
    pip install tensorflow==1.14.0 --user && \
    pip install ml_metrics==0.1.4 && \
    pip install --upgrade scipy==1.1.0 \
    pip uninstall -y numpy \
    pip install numpy==1.16.4 \
    pip uninstall -y gast \
    pip install gast==0.2.2

WORKDIR /workdir
RUN git clone https://github.com/mack-the-psych/vdok3.git
