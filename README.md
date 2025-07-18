# Feature Subspace Transfer Deep Matrix Factorization (FST‑DMF)

## 📜 Paper in one sentence

**FST‑DMF** augments a target rating matrix with an **auxiliary view of the items** — either

1. **side attributes** of the same dataset (e.g. one‑hot year & genre) or
2. **an entire second rating matrix** (e.g. MovieLens‑1M when the target is MovieLens‑100K).

That auxiliary matrix is first converted into an orthonormal basis `V_A`. The model then makes use of it in two complementary ways:

* **V‑init.** A *latent matrix* `ε` produced by a semi‑autoencoder (trained on the target rating matrix) is \*\*copied into the item embedding table \*\***`V`** before training.
* **Sub‑space loss.** A projection term keeps the learned item factors inside (or close to) the sub‑space spanned by `V_A`, so the knowledge encoded in the auxiliary view is preserved throughout training.
assets/1.png
---

## 🔧 Building the auxiliary matrix `V_A`

| **Source of auxiliary info**                                                                                           | Pre‑processing pipeline                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Item attributes** (e.g. year, genre) from the **same** dataset                                                       | 1️⃣ One‑hot encode every categorical slot → matrix `C` 2️⃣ Thin‑QR/SVD → orthonormal basis `V_A` with `V_A^T V_A = I`.                                                       |
| **Ratings of the same titles in a **********************************second********************************** dataset** | 1️⃣ Keep only items present in *both* domains 2️⃣ Feed the source rating matrix to a **semi‑autoencoder**; grab the hidden layer ε 3️⃣ Thin‑QR/SVD → orthonormalise → `V_A`. |

At training time we never change `V_A`; it stays fixed while the target‑domain item matrix `V` is **(i) initialised from it and (ii) nudged back toward its sub‑space** by the projection loss.
assets/1.png

---

## Why "plain" Deep Matrix Factorization (DMF) is not enough (DMF) is not enough

| Challenge                        | What happens with plain DMF?                                                                                                                   |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| *Data sparsity* in small domains | DMF learns item/user embeddings from scratch; with only a few ratings it easily **over‑fits**.                                                 |
| *Cold‑start across domains*      | Items that exist in both MovieLens‑1M and MovieLens‑100K end up with **incompatible embeddings**, so knowledge in the larger domain is wasted. |
| *Latent space drift*             | Without additional constraints, the learned space may rotate/scale unpredictably between runs, making cross‑domain comparison hard.            |

FST‑DMF answers all three issues: it **initialises** the item matrix with a well‑shaped latent space obtained from the big domain and **locks** that space during fine‑tuning.

---

## The two key innovations

### 1. `V‑init` – latent seed from a semi‑autoencoder

1. Feed the chosen rating matrix (target domain (MovieLens‑100K)) into a *semi‑autoencoder*.
2. Take the hidden layer output `ε ∈ ℝ^{n×d}` **without orthogonalising** it.
3. Copy each row of `ε` into the corresponding row of the item‑factor matrix `V` as the starting point for DMF.

### 2. Sub‑space consistency loss

During training on MovieLens‑100K we add

$$
\mathcal{L}_{\text{sub}} = -\frac{\alpha\,\eta}{2n}\,\bigl\| V_A V_A^{\top} V \bigr\|_F^{2}
$$

which maximises the alignment between the learned item matrix $V$ and the fixed sub‑space spanned by $V_A$.

* High $\eta$ → stronger pull when the two datasets are very similar.
* $\alpha$ comes from the paper’s hyper‑parameter grid.

---

## Complete loss function

$$
\mathcal{L} = \underbrace{\frac12\| (\hat R - R) \odot M \|_F^2}_{\text{reconstruction}}
+ \underbrace{\mathcal{L}_{\text{sub}}}_{\text{sub‑space regulariser}}
+ \underbrace{\frac{\alpha}{2n}\|V\|_F^2 + \frac{\beta}{2}\sum_i \|\theta_i\|_2^2}_{\text{weight decay}}
$$

* **$R$** – true rating matrix; **$\hat R$** – model prediction.
* \*\*Mask \*\***$M)$** filters out missing ratings so only observed entries contribute.
* **$V$** – trainable item factors; **$\theta_i$** – all other network weights.
* **$V_A$** is *fixed* and only appears inside $\mathcal{L}_{\text{sub}}$.

---

## 🔧 Hyper‑parameters

> The confidence parameter \$\eta\$ is fixed to **1** as recommended by the authors. Grid‑search therefore explores only the four remaining hyper‑parameters from Table 2.

| Hyper‑parameter       | Symbol     | Values searched |
| --------------------- | ---------- | --------------- |
| learning rate         | –          | 0.0123, 0.00123 |
| latent dimension      | \$d\$      | 100, 300        |
| item weight‑decay     | \$\alpha\$ | 0.10, 0.20      |
| network weight‑decay  | \$\beta\$  | 0.15, 0.40      |
| projection confidence | \$\eta\$   | **1 (fixed)**   |

Only the **16** combinations of the variable hyper‑parameters are evaluated for each train/val/test split; the best set is selected by the lowest RMSE on the validation set.

## Repository structure

```
├── colab_notebook.ipynb   # one‑click run & grid‑search in Google Colab
└── README.md              # this file 😄
```

> All code lives inside the notebook itself; there is no separate `src/` directory in this reproduction.
