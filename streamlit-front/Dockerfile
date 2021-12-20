FROM python:3.7-slim

RUN mkdir /streamlit-front

RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[server]\n\
port=8501\n\
headless=true\n\
[browser]\n\
serverAddress=localhost\n\
serverPort=8501\n\
" > /root/.streamlit/config.toml'

COPY requirements.txt /streamlit-front

WORKDIR /streamlit-front

RUN pip install -r requirements.txt

COPY . /streamlit-front

EXPOSE 8501

CMD ["streamlit", "run", "MOOui.py"]