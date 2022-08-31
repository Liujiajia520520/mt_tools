import torch
A = torch.randn(10, 13)
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
print(U.shape, S.shape, Vh.shape)
print(S)

U, S, Vh = torch.linalg.svd(A)
print(U.shape, S.shape, Vh.shape)
print(S)

U, S, Vh = torch.svd_lowrank(A)
print(U.shape, S.shape, Vh.shape)
print(S)

U, S, Vh = torch.pca_lowrank(A)
print(S)

