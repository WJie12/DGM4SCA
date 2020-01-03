from dgm4sca.dataset.loom import LoomDataset
from dgm4sca.models.scanvi import SCANVI
from dgm4sca.inference.annotation import compute_accuracy_rf, compute_accuracy_svc
from dgm4sca.inference.annotation import JointSemiSupervisedTrainer
from timeit import default_timer as timer
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

"""data loading"""
gene_dataset = LoomDataset(filename='simulation_3.loom', save_path="./data")
# gene_dataset = LoomDataset(filename='data_loom.loom', save_path="./data")
print(gene_dataset.labels)
print(gene_dataset)

use_batches = False
use_cuda = True
n_epochs = 20
n_cl = 10
scanvi = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels)
trainer = JointSemiSupervisedTrainer(scanvi, gene_dataset,
                                     n_labelled_samples_per_class=n_cl,
                                     classification_ratio=100, frequency=1)
trainer.labelled_set.to_monitor = ['reconstruction_error', 'accuracy']
trainer.unlabelled_set.to_monitor = ['reconstruction_error', 'accuracy']
tic1 = timer()
trainer.train(n_epochs=n_epochs)

accuracy = trainer.unlabelled_set.accuracy()
print(accuracy)

toc1 = timer()
print("DGM4SCA time:", toc1 - tic1)

""" Figure"""
accuracy_labelled_set = trainer.history["accuracy_labelled_set"]
accuracy_unlabelled_set = trainer.history["accuracy_unlabelled_set"]
print(accuracy_labelled_set, accuracy_unlabelled_set)
x = np.linspace(0, n_epochs, (len(accuracy_labelled_set)))
plt.plot(x, accuracy_labelled_set, label="accuracy labelled")
plt.plot(x, accuracy_unlabelled_set, label="accuracy unlabelled")
plt.legend()
plt.title('labelled VS unlabelled: accuracy')
plt.savefig("./fig/accuracy.png")
plt.clf()

reconstruction_error_labelled_set = trainer.history["reconstruction_error_labelled_set"]
reconstruction_error_unlabelled_set = trainer.history["reconstruction_error_unlabelled_set"]
print(reconstruction_error_labelled_set, reconstruction_error_unlabelled_set)
x = np.linspace(0, n_epochs, (len(reconstruction_error_labelled_set)))
plt.plot(x, reconstruction_error_labelled_set, label="reconstruction_error labelled")
plt.plot(x, reconstruction_error_unlabelled_set, label="reconstruction_error unlabelled")
plt.legend()
plt.title('labelled VS unlabelled: reconstruction_error')
plt.savefig("./fig/reconstruction_error.png")

############ baseline #########################
data_train, labels_train = trainer.labelled_set.raw_data()
data_test, labels_test = trainer.unlabelled_set.raw_data()

print(len(data_train))
print(len(labels_train))
print(len(data_test))
print(len(labels_test))

tic2 = timer()
svc_scores = compute_accuracy_svc(data_train, labels_train, data_test, labels_test)
toc2 = timer()
print("SVM time:", toc2 - tic2)
print("\nSVC score test :\n", svc_scores[0][1])
#
# tic3 = timer()
# rf_scores = compute_accuracy_rf(data_train, labels_train, data_test, labels_test)
# toc3 = timer()
# print("RF time:", toc3 - tic3)
# print("\nRF score train :\n", rf_scores[0][1])

