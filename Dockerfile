FROM python:3.8-buster

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

COPY NLPmoviereviews /NLPmoviereviews
COPY app.py /app.py
COPY setup.py /setup.py
COPY scripts /scripts

COPY saved_model /saved_model

RUN pip install .

CMD streamlit run app.py