FROM python:3.12.7-slim

WORKDIR /app

COPY MLProject/diabetes_preprocessing/ /app/diabetes_preprocessing/
COPY MLProject/modelling.py /app/modelling.py

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    dagshub

EXPOSE 5002

CMD ["mlflow", "models", "serve", "--host", "0.0.0.0", "--port", "5002"]