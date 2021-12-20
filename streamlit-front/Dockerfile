FROM python:3.7-slim

RUN mkdir -p /root/.streamlit
COPY config.toml /root/.streamlit

RUN mkdir /streamlit-front
COPY requirements.txt /streamlit-front

WORKDIR /streamlit-front

RUN pip install -r requirements.txt

COPY . /streamlit-front

EXPOSE 8501

CMD ["streamlit", "run", "MOOui.py"]