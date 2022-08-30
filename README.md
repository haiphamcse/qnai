QHD_HCM - Challenge 2 Quy Nhon AI 2022
======
_@_ Nguyen Ho Quang, Pham Duc Hai, Nguyen Ngoc Hai Dang.
This repo contains code for training and deploying our solution for QNAI Challenge 2 Competition 2022.

Installing
------
```python
cd submit
pip install -r requirements.txt
```
Training
------
```python
cd pytorch
python train.py --save_model "bert.pt" --train_path "data_train.csv" --test_path "data_test.csv"
```

Inferencing
------
```python
cd submit
python app.py
```
You can submit via POST at: http://127.0.0.1:6000/review-solver/solve?review_sentence
Add sentence you want to detect via passing as value to review_sentence
