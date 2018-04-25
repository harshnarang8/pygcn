from pandas import read_csv
from csv import QUOTE_NONE
import spacy
from tqdm import tqdm
import torch
import torchtext.vocab as vocab
import numpy as np
# from train import MAX_LEN, EMBEDDING_SIZE

MAX_LEN = 40
EMBEDDING_SIZE = 300

nlp = spacy.load('en')
def loadData(path):
  myDataset = read_csv(path, sep='\t', usecols=[4,5,6], header=None, quoting=QUOTE_NONE)
  print(len(myDataset))


  target_scores = myDataset[4].tolist()
  target_scores = [float(i) for i in target_scores]
  sentences1 =  myDataset[5].tolist()
  sentences2 =  myDataset[6].tolist()
  adj1 = []
  adj2 = []
  # print(sentences1[0])
  # print(sentences2[0])

  nlp = spacy.load('en')
  for idx, val in tqdm(enumerate(sentences1)):
    sentences1[idx], temp = createGraph(val)
    adj1.append(temp)
    # break

  for idx, val in tqdm(enumerate(sentences2)):
    sentences2[idx], tmp = createGraph(val)
    adj2.append(tmp)
    # break

  # sentences loaded
  return adj1, adj2, sentences1, sentences2, torch.FloatTensor(target_scores)


#loading glove embeddings
glove = vocab.GloVe(name='6B', dim=EMBEDDING_SIZE)
print('Loaded {} words'.format(len(glove.itos)))

def get_word(word):
  try:
    return glove.vectors[glove.stoi[word]]
  except KeyError:
    return torch.zeros(EMBEDDING_SIZE)

#testing for first sentence

# doc = sentences1[0]
'''
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion.')
print(doc)
nodeLabel1 = {}
for element in doc:
  nodeLabel1[element.text] = len(nodeLabel1)
  if element.has_vector:
    print(element.text)
    print(get_word(element.text.lower()))
print(nodeLabel1)
myGraph = torch.zeros((12, 12))


for token in doc:
  myGraph[nodeLabel1[token.head.text], nodeLabel1[token.orth_]] = 1 
  print(nodeLabel1[token.orth_]," *Child of* " , nodeLabel1[token.head.text])
print(myGraph)
'''

def createGraph(sentence): # works
  doc = nlp(sentence)
  # tempGraph = np.array((MAX_LEN, MAX_LEN))
  graph = []
  nodeMatrix = torch.zeros(len(doc), EMBEDDING_SIZE)
  nodes = {}
  for elem in doc:
    nodes[elem.text] = len(nodes)
    # if elem.has_vector:
    nodeMatrix[nodes[elem.text]] = get_word(elem.text.lower())
  
  for elem in doc:
    graph.append([nodes[elem.head.text], nodes[elem.orth_]])

  myGraph = torch.sparse.FloatTensor(torch.LongTensor(graph).t(), torch.ones(len(graph)), torch.Size([len(doc),len(doc)]))
  
  return (nodeMatrix, myGraph)

# print(createGraph(u'Apple is looking at buying U.K. startup for $1 billion.'))
# print((get_word(".")).size())
# print(torch.zeros(100).size())
loadData('../../stsbenchmark/sts-train.csv')
