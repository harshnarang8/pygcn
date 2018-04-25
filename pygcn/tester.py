from scipy.stats import pearsonr

x = []
y = []

with open('file.txt','r') as fp:
  for line in fp:
    x.append(float(line))

with open('file2.txt','r') as fp:
  for line in fp:
    y.append(float(line))

print(pearsonr(x, y))
