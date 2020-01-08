from dgm4sca.dataset.loom import LoomDataset
from dgm4sca.models.scanvi import SCANVI
from dgm4sca.inference.annotation import compute_accuracy_rf, compute_accuracy_svc
from dgm4sca.inference.annotation import JointSemiSupervisedTrainer
import matplotlib.pyplot as plt
import numpy as np
import argparse
import scanpy as sc
import anndata
import dgm4sca.utils.globalvar as gl
import warnings
import datetime

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='DGM4SCA')
parser.add_argument('-f', type=str, default='simulation_3.loom', help='datafile to use (default: data_loom.loom')
parser.add_argument('-e', type=int, default=100, help='epoch to run (default: 100')
parser.add_argument('-n', type=int, default=10, help='number of labelled cells for each label (default: 10')
parser.add_argument('-p', type=str, default='y', help='plot figures(default: y')
parser.add_argument('-t', type=int, default=1, help='test time(default: 1')
args = parser.parse_args()
filename = args.f
use_batches = False
use_cuda = True
n_epochs = args.e
n_cl = args.n
if args.p == "y":
    p = True
elif args.p == "n":
    p = False
else:
    p = True

test_time = args.t
print("## PRAMS to use:")
print("datafile:", filename)
print("epoch :", n_epochs)
print("labelled cells each label:", n_cl)
print("plot figures:", p)
print("test time to run:", test_time)
gl._init()

print("\n## STRAT: data loading")
gene_dataset = LoomDataset(filename=filename, save_path="./data")
print("cell type:", max(gene_dataset.labels))
print("gene_dataset:", gene_dataset)

for t in range(test_time):
    print("\n## Test time:", t)
    print("\n## STRAT: DGM4SCA")
    gl.set_value('mode', 'dgm4sca')
    scanvi = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels)

    trainer = JointSemiSupervisedTrainer(scanvi, gene_dataset,
                                         n_labelled_samples_per_class=n_cl,
                                         classification_ratio=100, frequency=1)
    trainer.labelled_set.to_monitor = ['reconstruction_error', 'accuracy']
    trainer.unlabelled_set.to_monitor = ['reconstruction_error', 'accuracy']
    tic1 = datetime.datetime.now()
    trainer.train(n_epochs=n_epochs)
    dgm4sca_accuracy = trainer.unlabelled_set.accuracy()
    print("DGM4SCA accuracy:", dgm4sca_accuracy)
    toc1 = datetime.datetime.now()
    dgm4sca_time = toc1 - tic1
    print("DGM4SCA time:", dgm4sca_time)

    if p:
        """ Figure"""
        accuracy_labelled_set = trainer.history["accuracy_labelled_set"]
        accuracy_unlabelled_set = trainer.history["accuracy_unlabelled_set"]
        x = np.linspace(0, n_epochs, (len(accuracy_labelled_set)))
        plt.plot(x, accuracy_labelled_set, label="accuracy labelled")
        plt.plot(x, accuracy_unlabelled_set, label="accuracy unlabelled")
        plt.legend()
        plt.title('labelled VS unlabelled: accuracy')
        plt.savefig("./figures/dgm4sca_accuracy.png")
        plt.clf()

        reconstruction_error_labelled_set = trainer.history["reconstruction_error_labelled_set"]
        reconstruction_error_unlabelled_set = trainer.history["reconstruction_error_unlabelled_set"]
        print(reconstruction_error_labelled_set, reconstruction_error_unlabelled_set)
        x = np.linspace(0, n_epochs, (len(reconstruction_error_labelled_set)))
        plt.plot(x, reconstruction_error_labelled_set, label="reconstruction_error labelled")
        plt.plot(x, reconstruction_error_unlabelled_set, label="reconstruction_error unlabelled")
        plt.legend()
        plt.title('labelled VS unlabelled: reconstruction_error')
        plt.savefig("./figures/dgm4sca_reconstruction_error.png")
        plt.clf()

        """ Latent Space Figure"""
        full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
        latent, _, labels = full.sequential().get_latent()
        post_adata = anndata.AnnData(X=gene_dataset.X)
        post_adata.obsm["X_scVI"] = latent
        if filename == "simulation_3.loom":
            post_adata.obs['cell_type'] = np.array([gene_dataset.cell_types[gene_dataset.labels[i][0]] for i in range(post_adata.n_obs)])
        else:
            post_adata.obs['cell_type'] = np.array([gene_dataset.CellTypes[gene_dataset.labels[i][0]] for i in range(post_adata.n_obs)])
        sc.pp.neighbors(post_adata, use_rep="X_scVI", n_neighbors=15)
        sc.tl.umap(post_adata, min_dist=0.1)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc.pl.umap(post_adata, color=["cell_type"], ax=ax, show=True, save='DGM4SCA_latent_space.png')

    print("\n## START: SCANVI")
    gl.set_value('mode', 'scanvi')
    scanvi = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels)
    trainer = JointSemiSupervisedTrainer(scanvi, gene_dataset,
                                         n_labelled_samples_per_class=n_cl,
                                         classification_ratio=100, frequency=1)
    trainer.labelled_set.to_monitor = ['reconstruction_error', 'accuracy']
    trainer.unlabelled_set.to_monitor = ['reconstruction_error', 'accuracy']
    tic1 = datetime.datetime.now()
    trainer.train(n_epochs=n_epochs)
    scanvi_accuracy = trainer.unlabelled_set.accuracy()
    print(scanvi_accuracy)
    toc1 = datetime.datetime.now()
    scanvi_time = toc1 - tic1
    print("SCANVI time:", scanvi_time)

    if p:
        """ Figure"""
        accuracy_labelled_set = trainer.history["accuracy_labelled_set"]
        accuracy_unlabelled_set = trainer.history["accuracy_unlabelled_set"]
        x = np.linspace(0, n_epochs, (len(accuracy_labelled_set)))
        plt.plot(x, accuracy_labelled_set, label="accuracy labelled")
        plt.plot(x, accuracy_unlabelled_set, label="accuracy unlabelled")
        plt.legend()
        plt.title('labelled VS unlabelled: accuracy')
        plt.savefig("./figures/scanvi_accuracy.png")
        plt.clf()

        reconstruction_error_labelled_set = trainer.history["reconstruction_error_labelled_set"]
        reconstruction_error_unlabelled_set = trainer.history["reconstruction_error_unlabelled_set"]
        x = np.linspace(0, n_epochs, (len(reconstruction_error_labelled_set)))
        plt.plot(x, reconstruction_error_labelled_set, label="reconstruction_error labelled")
        plt.plot(x, reconstruction_error_unlabelled_set, label="reconstruction_error unlabelled")
        plt.legend()
        plt.title('labelled VS unlabelled: reconstruction_error')
        plt.savefig("./figures/scanvi_reconstruction_error.png")
        plt.clf()

        """ Latent Space Figure"""
        full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
        latent, _, labels = full.sequential().get_latent()
        post_adata = anndata.AnnData(X=gene_dataset.X)
        post_adata.obsm["X_scVI"] = latent
        if filename == "simulation_3.loom":
            post_adata.obs['cell_type'] = np.array([gene_dataset.cell_types[gene_dataset.labels[i][0]] for i in range(post_adata.n_obs)])
        else:
            post_adata.obs['cell_type'] = np.array([gene_dataset.CellTypes[gene_dataset.labels[i][0]] for i in range(post_adata.n_obs)])
        sc.pp.neighbors(post_adata, use_rep="X_scVI", n_neighbors=15)
        sc.tl.umap(post_adata, min_dist=0.1)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc.pl.umap(post_adata, color=["cell_type"], ax=ax, show=True, save='SCANVI_latent_space.png')

    print("\n## START: SVM")
    data_train, labels_train = trainer.labelled_set.raw_data()
    data_test, labels_test = trainer.unlabelled_set.raw_data()
    # print(data_train)
    # print(data_test)

    print("Training dataset size: ", len(data_train))
    # print(len(labels_train))
    print("Testing dataset size: ", len(data_test))
    # print(len(labels_test))

    tic2 = datetime.datetime.now()
    svc_scores = compute_accuracy_svc(data_train, labels_train, data_test, labels_test)
    toc2 = datetime.datetime.now()
    svm_time = toc2-tic2
    svm_accuracy = svc_scores[0][1][1]
    print("SVM time:", svm_time)
    print("SVC score test :\n", svc_scores[0][1])

    # accuracy = "Accuracy\t"+str(dgm4sca_accuracy)+"\t"+str(scanvi_accuracy)+"\t"+ str(svm_accuracy)
    # time = "Time\t"+str(dgm4sca_time)+"\t"+str(scanvi_time)+"\t"+ str(svm_time)
    # result2txt = accuracy + "\n" + time
    # log_file = filename.split(".")[0] + "_e" + str(n_epochs) + "_n" + str(n_cl) + ".log"
    # with open(log_file, 'a') as file_handle:
    #     file_handle.write(result2txt)
    #     file_handle.write('\n')


# tic3 = timer()
# rf_scores = compute_accuracy_rf(data_train, labels_train, data_test, labels_test)
# toc3 = timer()
# print("RF time:", toc3 - tic3)
# print("\nRF score train :\n", rf_scores[0][1])

