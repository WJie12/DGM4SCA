from typing import Sequence
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson, kl_divergence as kl

from dgm4sca.models.log_likelihood import log_zinb_positive, log_nb_positive
from dgm4sca.models.modules import Encoder, DecoderSCVI
from dgm4sca.models.utils import one_hot

from dgm4sca.models.modules import Encoder, DecoderSCVI


torch.backends.cudnn.benchmark = True
import numpy as np
import torch
from torch.distributions import Normal, Categorical, kl_divergence as kl

from dgm4sca.models.classifier import Classifier
from dgm4sca.models.modules import Decoder, Encoder
from dgm4sca.models.utils import broadcast_labels



class SCANVI(nn.Module):
    r"""A semi-supervised Variational auto-encoder model - inspired from M1 + M2 model,
    as described in (https://arxiv.org/pdf/1406.5298.pdf). SCANVI stands for single-cell annotation using
    variational inference.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    :param y_prior: If None, initialized to uniform probability over cell types
    :param labels_groups: Label group designations
    :param use_labels_groups: Whether to use the label groups

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> scanvi = SCANVI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> gene_dataset = SyntheticDataset(n_labels=3)
        >>> scanvi = SCANVI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=3, y_prior=torch.tensor([[0.1,0.5,0.4]]), labels_groups=[0,0,1])
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        y_prior=None,
        labels_groups: Sequence[int] = None,
        use_labels_groups: bool = False,
        classifier_parameters: dict = dict(),
    ):
        super().__init__()

        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    #     super().__init__(
    #         n_input,
    #         n_hidden=n_hidden,
    #         n_latent=n_latent,
    #         n_layers=n_layers,
    #         dropout_rate=dropout_rate,
    #         n_batch=n_batch,
    #         dispersion=dispersion,
    #         log_variational=log_variational,
    #         reconstruction_loss=reconstruction_loss,
    #     )


        self.n_labels = n_labels
        # Classifier takes n_latent as input
        cls_parameters = {
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "dropout_rate": dropout_rate,
        }
        cls_parameters.update(classifier_parameters)
        self.classifier = Classifier(n_latent, n_labels=n_labels, **cls_parameters)

        self.encoder_z2_z1 = Encoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.decoder_z1_z2 = Decoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

        self.y_prior = torch.nn.Parameter(
            y_prior
            if y_prior is not None
            else (1 / n_labels) * torch.ones(1, n_labels),
            requires_grad=False,
        )
        self.use_labels_groups = use_labels_groups
        self.labels_groups = (
            np.array(labels_groups) if labels_groups is not None else None
        )
        if self.use_labels_groups:
            assert labels_groups is not None, "Specify label groups"
            unique_groups = np.unique(self.labels_groups)
            self.n_groups = len(unique_groups)
            assert (unique_groups == np.arange(self.n_groups)).all()
            self.classifier_groups = Classifier(
                n_latent, n_hidden, self.n_groups, n_layers, dropout_rate
            )
            self.groups_index = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.tensor(
                            (self.labels_groups == i).astype(np.uint8),
                            dtype=torch.uint8,
                        ),
                        requires_grad=False,
                    )
                    for i in range(self.n_groups)
                ]
            )

    def classify(self, x):
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, _, z = self.z_encoder(x)
        z = qz_m  # We classify using the inferred mean parameter of z_1 in the latent space
        if self.use_labels_groups:
            w_g = self.classifier_groups(z)
            unw_y = self.classifier(z)
            w_y = torch.zeros_like(unw_y)
            for i, group_index in enumerate(self.groups_index):
                unw_y_g = unw_y[:, group_index]
                w_y[:, group_index] = unw_y_g / (
                    unw_y_g.sum(dim=-1, keepdim=True) + 1e-8
                )
                w_y[:, group_index] *= w_g[:, [i]]
        else:
            w_y = self.classifier(z)
        return w_y

    def get_latents(self, x, y=None):
        zs = self.get_latents_vae(x)
        qz2_m, qz2_v, z2 = self.encoder_z2_z1(zs[0], y)
        if not self.training:
            z2 = qz2_m
        return [zs[0], z2]

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        is_labelled = False if y is None else True

        outputs = self.inference(x, batch_index, y)
        px_r = outputs["px_r"]
        px_rate = outputs["px_rate"]
        px_dropout = outputs["px_dropout"]
        qz1_m = outputs["qz_m"]
        qz1_v = outputs["qz_v"]
        z1 = outputs["z"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]

        # Enumerate choices of label
        ys, z1s = broadcast_labels(y, z1, n_broadcast=self.n_labels)
        qz2_m, qz2_v, z2 = self.encoder_z2_z1(z1s, ys)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        # KL Divergence
        mean = torch.zeros_like(qz2_m)
        scale = torch.ones_like(qz2_v)

        kl_divergence_z2 = kl(
            Normal(qz2_m, torch.sqrt(qz2_v)), Normal(mean, scale)
        ).sum(dim=1)
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = Normal(qz1_m, torch.sqrt(qz1_v)).log_prob(z1).sum(dim=-1)
        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)

        if is_labelled:
            return (
                reconst_loss + loss_z1_weight + loss_z1_unweight,
                kl_divergence_z2 + kl_divergence_l,
                0.0,
            )

        probs = self.classifier(z1)
        reconst_loss += loss_z1_weight + (
            (loss_z1_unweight).view(self.n_labels, -1).t() * probs
        ).sum(dim=1)

        kl_divergence = (kl_divergence_z2.view(self.n_labels, -1).t() * probs).sum(
            dim=1
        )
        kl_divergence += kl(
            Categorical(probs=probs),
            Categorical(probs=self.y_prior.repeat(probs.size(0), 1)),
        )
        kl_divergence += kl_divergence_l

        return reconst_loss, kl_divergence, 0.0
################### ADD FROM VAE.PY
    def get_latents_vae(self, x, y=None):
        r""" returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
            """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def inference(self, x, batch_index=None, y=None, n_samples=1, transform_batch=None):

        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            z = Normal(qz_m, qz_v.sqrt()).sample()
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = Normal(ql_m, ql_v.sqrt()).sample()

        if transform_batch is not None:
            dec_batch_index = transform_batch * torch.ones_like(batch_index)
        else:
            dec_batch_index = batch_index

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, z, library, dec_batch_index, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(dec_batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, **kwargs):
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout).sum(dim=-1)
        elif self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r).sum(dim=-1)
        elif self.reconstruction_loss == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss
