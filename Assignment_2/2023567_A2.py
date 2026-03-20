# Generated from: 2023567_A2.ipynb
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # SML A2


# Imports


import numpy as np
import matplotlib.pyplot as plt
import struct
import pandas as pd

# I have downloaded the Data directly from Kaggle [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset), hence no code block for the same.
# 


# #### Loading MNIST data


# raw binary files to numpy

def load_digit_data(images_path, labels_path):
    
    with open(labels_path, 'rb') as lb:
        _, num = struct.unpack(">II", lb.read(8))
        labels = np.fromfile(lb, dtype=np.uint8)

    with open(images_path, 'rb') as img:
        _, num, rows, cols = struct.unpack(">IIII", img.read(16))
        images = np.fromfile(img, dtype=np.uint8).reshape(num, rows, cols)

    return images, labels

X_train_all, y_train_all = load_digit_data(
    "Data/train-images.idx3-ubyte",
    "Data/train-labels.idx1-ubyte"
)
X_test_all, y_test_all = load_digit_data(
    "Data/t10k-images.idx3-ubyte",
    "Data/t10k-labels.idx1-ubyte"
)

print("X_train_all:", X_train_all.shape)
print("y_train_all:", y_train_all.shape)
print("X_test_all:", X_test_all.shape)
print("y_test_all:", y_test_all.shape)

np.random.seed(42)          # setting random seed for reproducibility

# Filter digits 0,1,2 
# 
# then
# 
# Sample 100 per Class (randomly)


classes=[0,1,2]
samples_per_class=100

def sample_digit_data(X,y):
    X_out, y_out = [], []
    for c in classes:
        idx_all = np.where(y == c)[0]
        idx100 = np.random.choice(idx_all, samples_per_class, replace=False)
        X_out.append(X[idx100])
        y_out.append(y[idx100])
    # img stacked vertically
    return np.vstack(X_out), np.hstack(y_out)

X_train, y_train = sample_digit_data(X_train_all, y_train_all)
X_test, y_test = sample_digit_data(X_test_all, y_test_all)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

plt.figure(figsize=(15, 15))

for i in range(X_train.shape[0]):
    plt.subplot(15, 20, i + 1)
    plt.imshow(X_train[i], cmap="gray")

    plt.text(
        1, 26, str(y_train[i]),
        color="white",
        fontsize=6,
        ha="left",
        va="bottom"
    )

    plt.axis("off")

plt.suptitle("Training Images", y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


plt.figure(figsize=(15, 15))

for i in range(X_test.shape[0]):
    plt.subplot(15, 20, i + 1)
    plt.imshow(X_test[i], cmap="gray")

    plt.text(
        1, 26, str(y_test[i]),
        color="white",
        fontsize=6,
        ha="left",
        va="bottom"
    )

    plt.axis("off")

plt.suptitle("Testing Images", y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Stacking the images


# Stack images from (N, 28, 28) to (N, 784) and transpose to (784, N)
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test  = X_test.reshape(X_test.shape[0], -1).T

# # Transpose labels from (N,) to (1,N)
# y_train = y_train.reshape(-1,1).T
# y_test = y_test.reshape(-1,1).T

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Normalizing the data 


X_train = X_train / 255.0
X_test = X_test / 255.0

# $\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i == y_i)$ 


def compute_accuracy(y_true, y_pred):
    N = len(y_true)
    return (1/N) * np.sum(y_true == y_pred)

# ##### Note: code till here was reused from my SML Assignment-1


# ### PCA
# 
# Mean of the data:
# 
# $\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$
# 
# Mean centered data:
# 
# $X_c = X - \mu$
# 
# Covariance matrix:
# 
# $S = \frac{X_c X_c^T}{N-1}$
# 
# Eigenvalue decomposition:
# 
# $S u_i = \lambda_i u_i$
# 
# Projection matrix using the top $p$ eigenvectors:
# 
# $U_p = [u_1, u_2, ..., u_p]$
# 
# Low dimensional representation:
# 
# $Y = U_p^T X_c$
# 
# Projection of a test sample:
# 
# $y_{test} = U_p^T (x_{test} - \mu)$
# 
# Reconstruction:
# 
# $\hat{x} = U_p Y + \mu$


def pca_fit_var(X, variance):
    
    d=X.shape[0]
    N=X.shape[1]

    mu = (1/N)* np.sum(X, axis=1, keepdims=True)
    Xc = X - mu

    # cov matrix
    S = (Xc @ Xc.T) / (N-1)

    eigvals, eigvecs = np.linalg.eigh(S)
    
    # sort in descending
    pairs = list(zip(eigvals, eigvecs.T))
    pairs.sort(reverse=True, key=lambda x: x[0])
    eigvals = [p[0] for p in pairs]
    eigvecs = np.array([p[1] for p in pairs]).T

    # find suitable p
    total_eigvals = sum(eigvals)

    cumsum_eigvals = 0
    p = 0
    for i, val in enumerate(eigvals):
        cumsum_eigvals += val
        if (cumsum_eigvals / total_eigvals) >= variance:
            p = i + 1
            break

    # top p eigvectors
    Up = eigvecs[:, :p]

    return {
        "Up": Up,
        "mu": mu,
        "components": p
    }
    
    
def pca_fit_comp(X, components):

    d=X.shape[0]
    N=X.shape[1]

    mu = (1/N)* np.sum(X, axis=1, keepdims=True)
    Xc = X - mu

    # cov matrix
    S = (Xc @ Xc.T) / (N-1)

    eigvals, eigvecs = np.linalg.eigh(S)
    
    # sort in descending
    pairs = list(zip(eigvals, eigvecs.T))
    pairs.sort(reverse=True, key=lambda x: x[0])
    eigvals = [p[0] for p in pairs]
    eigvecs = np.array([p[1] for p in pairs]).T

    p = components
    # top p eigvectors
    Up = eigvecs[:, :p]

    return {
        "Up": Up,
        "mu": mu,
    }

def pca_transform(X, model):
    
    Xc = X - model["mu"]
    X_transformed = model["Up"].T @ Xc
    return X_transformed

def pca_reconstruct(Y, model):
    
    X_reconstructed = model["Up"] @ Y + model["mu"]
    return X_reconstructed

# ### FDA
# 
# Global mean:
# 
# $\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$
# 
# Class mean:
# 
# $\mu_c = \frac{1}{N_c}\sum_{x_i \in C_c} x_i$
# 
# Between-class scatter matrix:
# 
# $S_B = \sum_c N_c(\mu_c - \mu)(\mu_c - \mu)^T$
# 
# Within-class scatter matrix:
# 
# $S_W = \sum_c \sum_{x_i \in C_c}(x_i - \mu_c)(x_i - \mu_c)^T$
# 
# Generalized eigenvalue problem:
# 
# $S_W^{\dagger} S_B w = \lambda w$
# 
# Projection matrix:
# 
# $W = [w_1, w_2, ..., w_{C-1}]$
# 
# Projection of the data:
# 
# $Z = W^T X$


def fda_fit(X, y, classes):
    
    d = X.shape[0]
    N = X.shape[1]
    
    mu = (1/N)* np.sum(X, axis=1, keepdims=True)

    # Between class scatter matrix
    SB = np.zeros((d,d))                   # init with dxd 0s
    # Within class scatter matrix   
    SW = np.zeros((d,d))                   # init with dxd 0s

    for c in classes:
        
        X_c = X[:, y == c]
        Nc = X_c.shape[1]
        
        mu_c = (1/Nc) * np.sum(X_c, axis=1, keepdims=True)
        
        SB+= Nc * (mu_c - mu) @ (mu_c - mu).T

        diff = X_c - mu_c
        SW+= diff @ diff.T

    # solving generalized eigenvalue problem
    eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(SW) @ SB)          # used pseudo-inverse due to singularity of SW 

    # sort in descending
    idx = np.argsort(eigvals)[::-1]
    # take top C-1 eigvecs
    # remove imaginary parts also
    W = np.real(eigvecs[:, idx[:len(classes)-1]])

    return {
        "W": W
    }   

def fda_transform(X, model):
    
    Z = model["W"].T @ X
    return Z

# ### LDA
# 
# Class mean:
# 
# $\mu_c = \frac{1}{N_c}\sum_{x_i \in C_c} x_i$
# 
# Class covariance matrix:
# 
# $\Sigma_c = \frac{1}{N_c}\sum_{x_i \in C_c}(x_i-\mu_c)(x_i-\mu_c)^T$
# 
# Shared covariance matrix:
# 
# $\Sigma = \frac{1}{C}\sum_{c=1}^{C} \Sigma_c$
# 
# Regularized covariance matrix:
# 
# $\Sigma_{reg} = \Sigma + \lambda I$
# 
# where $\lambda = 10^{-3}$
# 
# Weight vector:
# 
# $w_c = \Sigma_{reg}^{-1}\mu_c$
# 
# Linear discriminant function:
# 
# $g_c(x) = w_c^T x - \frac{1}{2}\mu_c^T \Sigma_{reg}^{-1}\mu_c$
# 
# Prediction rule:
# 
# $\hat{y} = \arg\max_c g_c(x)$


def lda_fit(X, y, classes, lamb=1e-3):
    
    means = {}
    covs = {}

    for c in classes:
        
        Xc = X[:, y == c]
        Nc = Xc.shape[1]
        mu = (1/Nc) * np.sum(Xc, axis=1)
        
        centered = Xc - mu.reshape(-1,1)
        # class specific cov
        cov = (1/Nc) * (centered @ centered.T) 
        
        means[c] = mu
        covs[c] = cov

    # shared cov matrix for all classes
    shared_cov = (1/len(classes)) * sum(covs.values())                                           
    shared_cov += lamb * np.eye(shared_cov.shape[0])        # regularizing
    shared_cov_inv = np.linalg.inv(shared_cov)

    return {
        "means": means,
        "cov_inv": shared_cov_inv,
        "classes": classes
    }

def lda_predict(X, model):
    
    lda_scores = []
    
    for c in model["classes"]:
        
        mu = model["means"][c]
        w = model["cov_inv"] @ mu
        g_c = w.T @ X - 0.5 * mu.T @ model["cov_inv"] @ mu
        lda_scores.append(g_c)                                        # lda score for class c

    return np.argmax(np.vstack(lda_scores), axis=0)                   # return c corresponding to max score

# ### QDA
# 
# Class mean:
# 
# $\mu_c = \frac{1}{N_c}\sum_{x_i \in C_c} x_i$
# 
# Class covariance matrix:
# 
# $\Sigma_c = \frac{1}{N_c}\sum_{x_i \in C_c}(x_i-\mu_c)(x_i-\mu_c)^T$
# 
# Regularized covariance matrix:
# 
# $\Sigma_{c,reg} = \Sigma_c + \lambda I$
# 
# where $\lambda = 10^{-3}$
# 
# Quadratic discriminant function:
# 
# $g_c(x) =
# -\frac{1}{2}(x-\mu_c)^T \Sigma_{c,reg}^{-1}(x-\mu_c)
# -\frac{1}{2}\log|\Sigma_{c,reg}|$
# 
# Prediction rule:
# 
# $\hat{y} = \arg\max_c g_c(x)$


def qda_fit(X, y, classes, lamb=1e-3):
    
    means = {}
    covs = {}
        
    for c in classes:
        
        Xc = X[:, y == c]
        Nc = Xc.shape[1]
        mu = (1/Nc) * np.sum(Xc, axis=1)
        
        centered = Xc - mu.reshape(-1,1)
        # class specific cov
        cov = (1/Nc) * (centered @ centered.T) 

        cov = cov + lamb * np.eye(cov.shape[0])        # regularizing
        
        means[c] = mu
        covs[c] = cov

    return {
        "means": means,
        "covs": covs,
        "classes": classes
    }

def qda_predict(X, model):
    
    qda_scores = []

    for c in model["classes"]:
        
        mu = model["means"][c]
        cov = model["covs"][c]

        cov_inv = np.linalg.inv(cov)
        sign, log_det = np.linalg.slogdet(cov)

        Xc = X - mu.reshape(-1,1)
        # quad_term = np.sum(Xc * (cov_inv @ Xc), axis=0)
        quad_term = np.diag(Xc.T @ cov_inv @ Xc)

        g_c = -0.5 * quad_term - 0.5 * log_det
        qda_scores.append(g_c)                                    # qda score for class c

    return np.argmax(np.vstack(qda_scores), axis=0)               # return c corresponding to max score

# # Experiments


results = []

# ## Experiment 0: LDA & QDA


lda_raw = lda_fit(X_train, y_train, classes)
qda_raw = qda_fit(X_train, y_train, classes)

lda_train = compute_accuracy(y_train, lda_predict(X_train, lda_raw))
lda_test  = compute_accuracy(y_test, lda_predict(X_test, lda_raw))

qda_train = compute_accuracy(y_train, qda_predict(X_train, qda_raw))
qda_test  = compute_accuracy(y_test, qda_predict(X_test, qda_raw))


print("LDA Train:", lda_train)
print("LDA Test:",  lda_test)

print("QDA Train:", qda_train)
print("QDA Test:",  qda_test)

results.append(["LDA", lda_train, lda_test])
results.append(["QDA", qda_train, qda_test])

# ## Experiment 1: FDA + LDA & QDA


fda_model = fda_fit(X_train, y_train, classes)

Xtr_fda = fda_transform(X_train, fda_model)
Xte_fda = fda_transform(X_test, fda_model)

lda_model = lda_fit(Xtr_fda, y_train, classes)
qda_model = qda_fit(Xtr_fda, y_train, classes)

fda_lda_train = compute_accuracy(y_train, lda_predict(Xtr_fda, lda_model))
fda_lda_test = compute_accuracy(y_test, lda_predict(Xte_fda, lda_model))

fda_qda_train = compute_accuracy(y_train, qda_predict(Xtr_fda, qda_model))
fda_qda_test = compute_accuracy(y_test, qda_predict(Xte_fda, qda_model))

print("FDA + LDA Train:", fda_lda_train)
print("FDA + LDA Test:",  fda_lda_test)

print("FDA + QDA Train:", fda_qda_train)
print("FDA + QDA Test:",  fda_qda_test)

results.append(["FDA + LDA",
                fda_lda_train,
                fda_lda_test])

results.append(["FDA + QDA",
                fda_qda_train,
                fda_qda_test])

# FDA 2D PLOT (TRAIN + TEST)


markers = ['o', 's', '^']                   # class 0,1,2

plt.figure(figsize=(12,5))

# Train
plt.subplot(1,2,1)

for i, c in enumerate(classes):
    idx = y_train == c
    plt.scatter(np.real(Xtr_fda[0, idx]),
                np.real(Xtr_fda[1, idx]),
                marker=markers[i],
                label=f"Class {c}",
                alpha=0.7)

plt.title("FDA Projection (Train)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.grid(True)


# Test
plt.subplot(1,2,2)

for i, c in enumerate(classes):
    idx = y_test == c
    plt.scatter(np.real(Xte_fda[0, idx]),
                np.real(Xte_fda[1, idx]),
                marker=markers[i],
                label=f"Class {c}",
                alpha=0.7)

plt.title("FDA Projection (Test)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ## Experiment 2: PCA 75% + LDA & QDA


pca_75 = pca_fit_var(X_train, variance=0.75)

Xtr_75 = pca_transform(X_train, pca_75)
Xte_75 = pca_transform(X_test, pca_75)

lda_model_75 = lda_fit(Xtr_75, y_train, classes)
qda_model_75 = qda_fit(Xtr_75, y_train, classes)

pca75_lda_train = compute_accuracy(y_train, lda_predict(Xtr_75, lda_model_75))
pca75_lda_test = compute_accuracy(y_test, lda_predict(Xte_75, lda_model_75))

pca75_qda_train = compute_accuracy(y_train, qda_predict(Xtr_75, qda_model_75))
pca75_qda_test = compute_accuracy(y_test, qda_predict(Xte_75, qda_model_75))

print("PCA75 + LDA Train:", pca75_lda_train)
print("PCA75 + LDA Test:",  pca75_lda_test)

print("PCA75 + QDA Train:", pca75_qda_train)
print("PCA75 + QDA Test:",  pca75_qda_test)

results.append(["PCA75 + LDA",
                pca75_lda_train,
                pca75_lda_test])

results.append(["PCA75 + QDA",
                pca75_qda_train,
                pca75_qda_test])

# RECONSTRUCTION + MSE (PCA 75%)


X_recon_75 = pca_reconstruct(Xtr_75, pca_75)                # Xtr_75 = pca_transform(X_train, pca_75)

mse = np.mean(np.square(X_recon_75-X_train))
print("Reconstruction MSE (75% PCA):", mse)

# Show 5 reconstructed training samples


plt.figure(figsize=(10,4))

for i in range(5):

    plt.subplot(2,5,i+1)
    plt.imshow(X_train[:, i].reshape(28,28), cmap='gray')
    plt.title(f"Original {i+1}")
    plt.axis("off")

    plt.subplot(2,5,5+i+1)
    plt.imshow(X_recon_75[:, i].reshape(28,28), cmap='gray')
    plt.title(f"Reconstructed {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ## Experiment 3: PCA 90% + LDA & QDA


pca_90 = pca_fit_var(X_train, variance=0.90)

Xtr_90 = pca_transform(X_train, pca_90)
Xte_90 = pca_transform(X_test, pca_90)

lda_model_90 = lda_fit(Xtr_90, y_train, classes)
qda_model_90 = qda_fit(Xtr_90, y_train, classes)

pca90_lda_train = compute_accuracy(y_train, lda_predict(Xtr_90, lda_model_90))
pca90_lda_test = compute_accuracy(y_test, lda_predict(Xte_90, lda_model_90))

pca90_qda_train = compute_accuracy(y_train, qda_predict(Xtr_90, qda_model_90))
pca90_qda_test = compute_accuracy(y_test, qda_predict(Xte_90, qda_model_90))

print("PCA90 + LDA Train:", pca90_lda_train)
print("PCA90 + LDA Test:",  pca90_lda_test)

print("PCA90 + QDA Train:", pca90_qda_train)
print("PCA90 + QDA Test:",  pca90_qda_test)

results.append(["PCA90 + LDA",
                pca90_lda_train,
                pca90_lda_test])

results.append(["PCA90 + QDA",
                pca90_qda_train,
                pca90_qda_test])

# RECONSTRUCTION + MSE (PCA 90%)


X_recon_90 = pca_reconstruct(Xtr_90, pca_90)                # Xtr_90 = pca_transform(X_train, pca_90)

mse = np.mean(np.square(X_recon_90-X_train))
print("Reconstruction MSE (90% PCA):", mse)

# Show 5 reconstructed training samples


plt.figure(figsize=(10,4))

for i in range(5):

    plt.subplot(2,5,i+1)
    plt.imshow(X_train[:, i].reshape(28,28), cmap='gray')
    plt.title(f"Original {i+1}")
    plt.axis("off")

    plt.subplot(2,5,5+i+1)
    plt.imshow(X_recon_90[:, i].reshape(28,28), cmap='gray')
    plt.title(f"Reconstructed {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ## Experiment 4: PCA 2 component + LDA & QDA


pca_2 = pca_fit_comp(X_train, components=2)

Xtr_2 = pca_transform(X_train, pca_2)
Xte_2 = pca_transform(X_test, pca_2)

lda_model_2 = lda_fit(Xtr_2, y_train, classes)
qda_model_2 = qda_fit(Xtr_2, y_train, classes)

pca2_lda_train = compute_accuracy(y_train, lda_predict(Xtr_2, lda_model_2))
pca2_lda_test = compute_accuracy(y_test, lda_predict(Xte_2, lda_model_2))

pca2_qda_train = compute_accuracy(y_train, qda_predict(Xtr_2, qda_model_2))
pca2_qda_test = compute_accuracy(y_test, qda_predict(Xte_2, qda_model_2))

print("PCA2 + LDA Train:", pca2_lda_train)
print("PCA2 + LDA Test:",  pca2_lda_test)

print("PCA2 + QDA Train:", pca2_qda_train)
print("PCA2 + QDA Test:",  pca2_qda_test)


results.append(["PCA2 + LDA",
                pca2_lda_train,
                pca2_lda_test])

results.append(["PCA2 + QDA",
                pca2_qda_train,
                pca2_qda_test])

# RECONSTRUCTION + MSE (PCA 90%)


X_recon_2 = pca_reconstruct(Xtr_2, pca_2)                # Xtr_90 = pca_transform(X_train, pca_90)

mse = np.mean(np.square(X_recon_2-X_train))
print("Reconstruction MSE (2 components):", mse)

# Show 5 reconstructed training samples


plt.figure(figsize=(10,4))

for i in range(5):

    plt.subplot(2,5,i+1)
    plt.imshow(X_train[:, i].reshape(28,28), cmap='gray')
    plt.title(f"Original {i+1}")
    plt.axis("off")

    plt.subplot(2,5,5+i+1)
    plt.imshow(X_recon_2[:, i].reshape(28,28), cmap='gray')
    plt.title(f"Reconstructed {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# PCA 2 COMPONENTS PLOT


markers = ['o', 's', '^']                   # class 0,1,2

plt.figure(figsize=(6,6))

for i, c in enumerate(classes):
    idx = y_train == c
    plt.scatter(Xtr_2[0, idx],
                Xtr_2[1, idx],
                marker=markers[i],
                label=f"Class {c}",
                alpha=0.7)

plt.title("PCA (First 2 Principal Components)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# Summary of results


results_table = pd.DataFrame(results, columns=["Method", "Train Accuracy", "Test Accuracy"])

results_table.insert(0, "Experiment", [i//2 for i in range(len(results_table))])
results_table = results_table.set_index("Experiment")

print(results_table)

# -- Umang Aggarwal (2023567)