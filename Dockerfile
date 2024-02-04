FROM python:3.8.18-slim
RUN ["mkdir", "/var/code"]
RUN ["mkdir", "/var/code/data"]
RUN ["mkdir", "/var/code/model"]
RUN ["mkdir", "/var/code/templates"]
COPY ./data/final_test.csv /var/code/data/final_test.csv
COPY ./templates/*.html /var/code/templates/
COPY ./requirements.txt /var/code/requirements.txt
COPY ./web_infer.py /var/code/web_infer.py
COPY ./training.py /var/code/training.py
WORKDIR /var/code
RUN pip install --no-cache-dir -r requirements.txt 
EXPOSE 8000
CMD ["python", "./web_infer.py"]

