FROM python:3.11

WORKDIR /insurance_prediction

COPY  requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "insurance_model/model.py"]
