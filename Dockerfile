# Requirements in addition to the code in the repo:
# 1) Pretrained <saved_models/> for SyntaxSQL should be placed in this directory
# 2) <glove/glove.42B.300d.txt> should be placed in this directory

FROM vanessa/pytorch-dev:py2

EXPOSE 6000

COPY . /workspace/syntaxSQL

WORKDIR /workspace/syntaxSQL
RUN ["pip", "install", "--upgrade", "pip"]
RUN ["pip", "install", "-r", "requirements.txt"]
RUN ["python", "nltk_download.py"]

# Mount dq-data volume here so Spider dataset is included!
VOLUME /workspace/data

ENTRYPOINT ["python", "main.py", "--config_path", "docker_cfg.ini"]
