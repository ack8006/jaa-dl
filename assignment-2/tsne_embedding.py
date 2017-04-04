import numpy as np
import pylab as Plot
import pickle as pkl
from tsne import tsne


matrix = pkl.load(open('./data/best_model_word_embeddings.pkl','rb'));
labels = pkl.load(open('./data/word_mapping.pkl','rb'))


reduced_matrix = tsne(matrix,2, max_iter=1000)


Plot.figure(figsize=(200, 200), dpi=100)
max_x = np.amax(reduced_matrix, axis=0)[0]
max_y = np.amax(reduced_matrix, axis=0)[1]
Plot.xlim((-max_x,max_x))
Plot.ylim((-max_y,max_y))


Plot.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);


for id in range(0, len(labels)):
    target_label = labels[id]
    x = reduced_matrix[id, 0]
    y = reduced_matrix[id, 1]
    Plot.annotate(target_label, (x,y))

Plot.savefig("tsne.png");

