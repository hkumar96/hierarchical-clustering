# Hierarchical-clustering

## Packages used:
1. stemming
2. numpy

## How to install packages?
```shell
sudo pip install -U <package_name>
```
## How to run the code?
```shell
python clustering.py
```

## Files in the folder
1. 'clustering.py' is the main python code
2. 'demo.csv' is a subset of main input dataset of first 1000 samples.
3. 'sms.csv' is the main input dataset
4. 'stopword.csv' is a dataset that contains stopwords in English language.

## Performance
1. Code takes 34.61s to run over the demo dataset

## Changing the input file
1. In the python code look for statement
```python
temp = read_csv("demo.csv")
```
2. Change the file name here.

## Changing the threshold
1. In the python code look for statement
```python
clustering(sim,0.7)
```
2. Change the value '0.7' to new threshold

## About the algorithm
1. Johnson's Agglomerative clustering algorithm has been used to form the hierarchical clusters in bottom up manner.
2. Output of the program are the clusters seperated by brackets.
3. Closest clusters are innermost ones and as we move outwards the similarity between the clusters decreases.
4. 'level(m)' displays the similarity between 'm' clusterings.
5. Uncomment the 'print (level)' to print the value of similarity between the levels.
