from dgm4sca.dataset.loom import LoomDataset
from dgm4sca.models.scanvi import SCANVI
from dgm4sca.inference.annotation import compute_accuracy_rf, compute_accuracy_svc
from dgm4sca.inference.annotation import JointSemiSupervisedTrainer
from timeit import default_timer as timer

"""data loading"""
gene_dataset = LoomDataset(filename='simulation_3.loom', save_path="./data")
print(gene_dataset)
# gene_dataset = LoomDataset(filename='data_loom.loom', save_path="./data")

use_batches=False
use_cuda=True
n_epochs_all=None
n_epochs = 10 if n_epochs_all is None else n_epochs_all
n_cl = 10
scanvi = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels)
trainer = JointSemiSupervisedTrainer(scanvi, gene_dataset,
                                     n_labelled_samples_per_class=n_cl,
                                     classification_ratio=100)

tic = timer()
trainer.train(n_epochs=n_epochs)
accuracy = trainer.unlabelled_set.accuracy()
toc = timer()
print(toc - tic)
print(accuracy)

############ baseline #########################
data_train, labels_train = trainer.labelled_set.raw_data()
data_test, labels_test = trainer.unlabelled_set.raw_data()

print(len(data_train))
print(len(labels_train))
print(len(data_test))
print(len(labels_test))

tic = timer()
svc_scores = compute_accuracy_svc(data_train, labels_train, data_test, labels_test)
toc = timer()
print(toc - tic)
print("\nSVC score test :\n", svc_scores[0][1])
#
# tic = timer()
# rf_scores = compute_accuracy_rf(data_train, labels_train, data_test, labels_test)
# toc = timer()
# print(toc - tic)
# print("\nRF score train :\n", rf_scores[0][1])

