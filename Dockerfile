FROM vanessa/pytorch-dev:py2

ARG models_path=saved_models/
ARG glove_path=glove/

EXPOSE 6000

COPY . /workspace/syntaxSQL

WORKDIR /workspace/syntaxSQL
RUN ["pip", "install", "--upgrade", "pip"]
RUN ["pip", "install", "-r", "requirements.txt"]
RUN ["python", "nltk_download.py"]
