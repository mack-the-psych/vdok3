# vdok3
## Python3-based ML library for vdok (vocabulary depth of knowledge) measurement

## Docker usage example
$ docker build . -t trial_vdok3 <br>
$ docker run -v /home/ubuntu/MELVA-S:/MELVA-S -p 9999:9999 -p 6006:6006 -it --name vdok3_run trial_vdok3 <br>
/workdir# jupyter notebook --port 9999 --ip=0.0.0.0 --allow-root <br>
