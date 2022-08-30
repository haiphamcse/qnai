FROM python:3.9.13

WORKDIR C:\New_DH\DucHai_Legacy\Duc Hai\O E Backup\PyCharm\PhoBert_1.1\submit
COPY . .

# STEP 4,5,6: INSTALL NECESSARY PACKAGE
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

#RUN gdown filedir
CMD ["python", "./app.py"]

