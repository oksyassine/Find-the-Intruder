#----------------------------Libraries---------------------------------#

from collections import Counter
import torch
import torch.optim as optim
import numpy as np
import math
from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.data.iterators import BasicIterator
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from allennlp.training.checkpointer import Checkpointer
from overrides import overrides
from scipy.stats import spearmanr,pearsonr
from torch.nn import CosineSimilarity
from torch.nn import functional

CUDA_DEVICE = 0
#----------------------Defining the DatasetReader and Preparing the data----------------------------------#

@DatasetReader.register("skip_gram")
class SkipGramReader(DatasetReader):
    def __init__(self, window_size=2, vocab: Vocabulary=None):
        #if we set "lazy" variable to True, DatasetReader will create and yield up instances as needed rather than all at once.
        super().__init__(lazy=False)
        #window_size is the maximum distance between the focus word and its contextual neighbors
        self.window_size = window_size
        self.reject_probs = None
        if vocab:
            self.reject_probs = {}#it will contain a pre-computed dict of probabilities for subsampling
            threshold = 6*1.e-3 # freq_max of rare Words not subsampled
            token_counts = vocab._retained_counter['token_in'] # collections.defaultdict # would be set only if instances were used for vocabulary construction
            total_counts = sum(token_counts.values())
            
            for _, token in vocab.get_index_to_token_vocabulary('token_in').items():
                counts = token_counts[token]#Count the number of times a word is reapeated in the text corpus
                if counts > 0:
                    normalized_counts = counts / total_counts
                    reject_prob = 1. - math.sqrt(threshold / normalized_counts)
                    reject_prob = max(0., reject_prob)
                else:
                    reject_prob = 0.
                self.reject_probs[token] = reject_prob

    def _subsample_tokens(self, tokens):
        """Given a list of tokens, runs sub-sampling.

        Returns a new list of tokens where rejected tokens are replaced by Nones.
        """
        new_tokens = []
        for token in tokens:
            reject_prob = self.reject_probs.get(token, 0.)
            if 0 < reject_prob:
                new_tokens.append(None)
            else:
                new_tokens.append(token)

        return new_tokens

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as text_file:
            for line in text_file:
                tokens = line.strip().split(' ') #list of tokens
                tokens = tokens[:9750000] # Due To limitation on the RAM Memory, we should truncate the training data first,
                                          # by using only the first nine million tokens
                if self.reject_probs:
                    tokens = self._subsample_tokens(tokens)
                    print(tokens[:200])  # for debugging

                for i, token in enumerate(tokens):
                    if token is None:
                        continue

                    token_in = LabelField(token, label_namespace='token_in')

                    for j in range(i - self.window_size, i + self.window_size + 1):
                        if j < 0 or i == j or j > len(tokens) - 1:
                            continue

                        if tokens[j] is None:
                            continue

                        token_out = LabelField(tokens[j], label_namespace='token_out') # Labelfield
                        yield Instance({'token_in': token_in, 'token_out': token_out}) # Instance
reader = SkipGramReader()
text8 = reader.read('data/text8')
#Once we've read the dataset, we use it to create our vocabulary
vocab = Vocabulary.from_instances(text8, min_count={'token_in': 2, 'token_out': 2})
del(text8)
#reading the dataset with Vocabulary to sub-sample the frequent words 
reader = SkipGramReader(vocab=vocab)
text8 = reader.read('data/text8')

BATCH_SIZE = 256 #batch_size specifies the size of the batch (the number of instances in a batch)
iterator = BasicIterator(batch_size=BATCH_SIZE)
iterator.index_with(vocab)
EMBEDDING_DIM = 300
embedding_in = Embedding(num_embeddings=vocab.get_vocab_size('token_in'),
                         embedding_dim=EMBEDDING_DIM)
if CUDA_DEVICE > -1:
  embedding_in = embedding_in.to(CUDA_DEVICE) # we are able now to use GPU 0 

#---------------------------Defining the skip-gram Model-----------------------------------#

#1 we implement the Skip-gram model
class SkipGramModel(Model):
    def __init__(self, vocab, embedding_in, cuda_device=-1):
        super().__init__(vocab)
#2 Embedding object is passed from outside rather than defined inside
        self.embedding_in = embedding_in
#3 this create a linear layer(we don't need biases)       
        self.linear = torch.nn.Linear(
            in_features=EMBEDDING_DIM,#size of the input vector
            out_features=vocab.get_vocab_size('token_out'),#size of the output vector
            bias=False)      
        if cuda_device > -1:
            self.linear = self.linear.to(cuda_device)
#4 Body of neural network computation is implemented in forward
    def forward(self, token_in, token_out):
#5 convert input tensors(word IDs) to word embeddings     
        embedded_in = self.embedding_in(token_in)
#6 Apply linear layer     
        logits = self.linear(embedded_in)
#7 This combines softmax and loss ompute the loss in a single function .c
        loss = functional.cross_entropy(logits, token_out)
        return {'loss': loss}
model = SkipGramModel(vocab=vocab,
                  embedding_in=embedding_in,
                  cuda_device=CUDA_DEVICE)

optimizer = optim.Adam(model.parameters())
checkpoint = Checkpointer(serialization_dir="checkpoint/")

#----------------------------training the model---------------------------------#

trainer = Trainer(model=model,
              optimizer=optimizer,
              iterator=iterator,
              train_dataset=text8,
              patience=3,
              num_epochs=30,
              checkpointer=checkpoint,
              cuda_device=CUDA_DEVICE)
trainer.train()

#----------------------------Evaluating and Exploring the Model Results---------------------------------#

def get_related(token: str, embedding: Model, vocab: Vocabulary, num_related: int = 20):
    """Given a token, return a list of top 20 most similar words to the token."""
    token_id = vocab.get_token_index(token, 'token_in')
    token_vec = embedding.weight[token_id]#A pre-initialization weight matrix for the embedding lookup, allowing the use of pretrained vectors.
    cosine = CosineSimilarity(dim=0)  #we do this to be able calculate simple cosine similarity between 2 vectors
    sims = Counter()

    for index, token in vocab.get_index_to_token_vocabulary('token_in').items():
        # Cosine similarity of our token vector with every other word vector in the vocabulary
        sim = cosine(token_vec, embedding.weight[index]).item() 
        sims[token] = sim #save the value of cosine similarity

    return sims.most_common(num_related)

print(get_related('december', embedding_in, vocab))
print(get_related('december', embedding_in, vocab))
print(get_related('december', embedding_in, vocab))


def read_simlex999():
    simlex999 = []
    with open('data/SimLex-999/SimLex-999.txt') as f: #open the test file SimLex-999.txt
        next(f)
        for line in f:
            fields = line.strip().split('\t') #there is a string who takes values from each line
            word1, word2, _, sim = fields[:4] #extract only the four first columns in the file SimLex-999.txt
            sim = float(sim) #cast value
            simlex999.append((word1, word2, sim)) #add each couple of words with their similarity into simlex999
    return simlex999

def evaluate_embeddings(embedding, vocab: Vocabulary):
    cosine = CosineSimilarity(dim=0)
    simlex999 = read_simlex999()
    sims_pred = []
    oov_count = 0
    for word1, word2, sim in simlex999:
        word1_id = vocab.get_token_index(word1, 'token_in') #word1_id takes the ID of the word 1.
        if word1_id == 1: # word_ID==1  means that that the word is out of vocabulary OOV
            sims_pred.append(0.)
            oov_count += 1 
            continue
        word2_id = vocab.get_token_index(word2, 'token_in') #word2_id takes the ID of the word 2
        if word2_id == 1:
            sims_pred.append(0.)
            oov_count += 1
            continue

        sim_pred = cosine(embedding.weight[word1_id],
                          embedding.weight[word2_id]).item() #Calculate the CosineSimilarity between word1 and word2 and charge this in sim_pred.
        sims_pred.append(sim_pred)

    assert len(sims_pred) == len(simlex999) # Assertion de l'egalité de longueur de sims_pred et simlex999
    print('# of OOV words: {} / {}'.format(oov_count, len(simlex999)))
    print(pearsonr(sims_pred, [sim for _, _, sim in simlex999]))
    return spearmanr(sims_pred, [sim for _, _, sim in simlex999]) # compare two sets of similarities and calculate how they are related, it's called spearman's correlation
    #compare two sets of similarities and calculate how they are related.
    #Calculates a Spearman rank-order correlation coefficient and the p-value to test for non-correlation.
    """scipy.stats.spearmanr(a, b=None, axis=0)[source]
Calculates a Spearman rank-order correlation coefficient and the p-value to test for non-correlation."""
    
rho = evaluate_embeddings(embedding_in, vocab)
print('simlex999 speareman correlation: {}'.format(rho))

#----------------------------Visualize The word embeddings---------------------------------#

# Reading a mixture of text and vectors
def read_model(file_path):
    with open(file_path) as f:
        i=0
        for line in f.readlines()[6:]:
            row = line.split()#for each line, split the line into :
            word = row[0] # word in row[0]
            vec = [float(x) for x in row[1:]]#and vector for the rest of the line in row[1:].
            yield (word, vec) #return every iteration ( yield : generator )
            i+=1#passing to the next ligne
            if i == 1100:
                break

from sklearn.manifold import TSNE #import TSNE technic from manifold
import matplotlib.pyplot as plt
words = []
vectors = []
for word, vec in read_model('embed17.txt'):#pour tous les 1100 couples de (word,vec) retourné
    words.append(word)#add each word to  words's array.
    vectors.append(vec)#parallèlement, ajouter chaque vecteur dans la liste vectors

modele = TSNE(n_components=2, init='pca', random_state=0) #Fixing Dimension of the embedded space et 2 ,PCA Initialization of embedding,the random number generator is the RandomState instance used by np.random.
coordinates = modele.fit_transform(vectors)#  Performs the calculations and returns the 2 principal components(Fit to data, then transform it.)

plt.figure(figsize=(50, 50))

for word, xy in zip(words, coordinates): #parallele aggregation of elements from two iterables(words,coordinates)
    plt.scatter(xy[0], xy[1])
    plt.annotate(word, #add annotations to our plot
                  xy=(xy[0], xy[1]),
                  xytext=(2, 2),
                  textcoords='offset points')

plt.xlim(-70, 70)
plt.ylim(-70, 70)
plt.show()

#----------------------------Export the model---------------------------------#

def write_embeddings(embedding: Embedding, file_path, vocab: Vocabulary):
    with open(file_path, mode='w') as f:
        words=vocab.get_index_to_token_vocabulary('token_in').items()
        print(len(words))
        f.write('{} {}\n'.format(len(words),EMBEDDING_DIM))#we write number of words and embedding dimension
        for index, token in words: #loop through both keys and values, by using the items()
            values = ['{:.10f}'.format(val) for val in embedding.weight[index]]#write each value as a number with 10 decimals
            f.write(' '.join([token] + values))
            f.write('\n')
            
write_embeddings(embedding_in,'embed17.txt', vocab)
