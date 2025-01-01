# Data preprocessing
Preprocessing steps to obtain query-specific graph $\widetilde{G}$

## Step 0: Download Freebase

```
wget https://download.microsoft.com/download/A/E/4/AE428B7A-9EF9-446C-85CF-D8ED0C9B1F26/FastRDFStore-data.zip --no-check-certificate
```

unzip `FastRDFStore-data.zip`.

## Step 1: Filter Freebase
```python
python get_id2name.py
python manual_filter_rel.py
python filter_rel.py
```

## Step 2: Prepare basic elements (entities, answers, topic entities) for dataset
```python
python preprocess_step0.py 
```

## Step 3: Use PPR algorithm to get $\widetilde{G}$ for each query
```python
python PPRmultithread.py
```
