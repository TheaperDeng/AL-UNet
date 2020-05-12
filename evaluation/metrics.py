import numpy as np


# ## Consider all four different cells:

# Unlabelled  =   np.array([0,     0,      0   ]) #black, background
# A           =   np.array([80,    0,      255 ]) #red, inflammatory cells 
# B           =   np.array([192,   255,    128 ]) #Light blue, nuclei
# C           =   np.array([64,    255,    64  ]) #green, cytoplasm
# M = [Unlabelled, A, B, C]


# # Input: ground truth image numpy matrix true_mat, 500*500*3.
# #        predicted image numpy matrix pred_mat, 500*500*3
# # Output: pixel accuracy, single value
# def pixel_acc(true_mat, pred_mat):
#     x, y = true_mat.shape[0], true_mat.shape[1]
#     sum_n_ii = 0
#     for i in range(x):
#         for j in range(y):
#             sum_n_ii += int((true_mat[i,j,:] == pred_mat[i,j,:]).all())

#     p_acc = sum_n_ii / (x*y)
#     print("The pixel accuracy is %f"%p_acc)
#     return p_acc


# # Input: ground truth image numpy matrix true_mat, 500*500*3.
# #        predicted image numpy matrix pred_mat, 500*500*3.
# #        classes list: M.
# # Output: mean pixel accuracy, single value
# def mean_pixel_acc(true_mat, pred_mat, M = M):
#     mean_p_acc = 0
#     x, y = true_mat.shape[0], true_mat.shape[1]
#     sum_n_ij = 0
#     num = 0
#     for m in M:
#         sum_true_m = 0
#         sum_n_ii_m = 0
#         for i in range(x):
#             for j in range(y):
#                 sum_true_m += int((true_mat[i,j,:] == m).all())
#                 if (true_mat[i,j,:] == m).all():
#                     sum_n_ii_m += int((true_mat[i,j,:] == pred_mat[i,j,:]).all())
#         if sum_true_m > 0:
#             sum_n_ij += sum_n_ii_m / sum_true_m
#             num += 1

#     mean_p_acc = sum_n_ij / (num + 1)
#     print("The mean pixel accuracy is %f"%mean_p_acc)
#     return mean_p_acc



# # Input: ground truth image numpy matrix true_mat, 500*500*3.
# #        predicted image numpy matrix pred_mat, 500*500*3.
# #        classes list: M.
# # Output: mean IOU, single value
# def mean_iou(true_mat, pred_mat, M = M):
#     mean_iou = 0
#     x, y = true_mat.shape[0], true_mat.shape[1]
#     sum_n_ij = 0
#     num = 0
#     for m in M:
#         sum_true_m = 0
#         sum_pred_m = 0
#         sum_n_ii_m = 0
#         for i in range(x):
#             for j in range(y):
#                 sum_true_m += int((true_mat[i,j,:] == m).all())
#                 sum_pred_m += int((pred_mat[i,j,:] == m).all())
#                 if (true_mat[i,j,:] == m).all():
#                     sum_n_ii_m += int((true_mat[i,j,:] == pred_mat[i,j,:]).all())
#         if (sum_true_m + sum_pred_m - sum_n_ii_m) > 0:
#             sum_n_ij += sum_n_ii_m / (sum_true_m + sum_pred_m - sum_n_ii_m)
#             num += 1

#     mean_iou = sum_n_ij / (num + 1)
#     print("The mean iou is %f"%mean_iou)
#     return mean_iou




## Consider only nucleis and inflammatory cells:

Unlabelled  =   np.array([0,     0,      0   ]) #black, background
A           =   np.array([80,    0,      255 ]) #red, inflammatory cells 
B           =   np.array([192,   255,    128 ]) #Light blue, nuclei
C           =   np.array([64,    255,    64  ]) #green, cytoplasm
M = [A, B, C]


# Input: ground truth image numpy matrix true_mat, 500*500*3.
#        predicted image numpy matrix pred_mat, 500*500*3
# Output: pixel accuracy, single value
def pixel_acc(true_mat, pred_mat):
    x, y = true_mat.shape[0], true_mat.shape[1]
    sum_n_ii = 0
    total_num = 0
    for i in range(x):
        for j in range(y):
            if (true_mat[i,j,:] == A).all() or (true_mat[i,j,:] == B).all() or (true_mat[i,j,:] == C).all():
                sum_n_ii += int((true_mat[i,j,:] == pred_mat[i,j,:]).all())
                total_num += 1

    p_acc = sum_n_ii / total_num
    print("The pixel accuracy is %f"%p_acc)
    return p_acc


# Input: ground truth image numpy matrix true_mat, 500*500*3.
#        predicted image numpy matrix pred_mat, 500*500*3.
#        classes list: M.
# Output: mean pixel accuracy, single value
def mean_pixel_acc(true_mat, pred_mat, M = M):
    mean_p_acc = 0
    x, y = true_mat.shape[0], true_mat.shape[1]
    sum_n_ij = 0
    num = 0
    for m in M:
        sum_true_m = 0
        sum_n_ii_m = 0
        for i in range(x):
            for j in range(y):
                sum_true_m += int((true_mat[i,j,:] == m).all())
                if (true_mat[i,j,:] == m).all():
                    sum_n_ii_m += int((true_mat[i,j,:] == pred_mat[i,j,:]).all())
        if sum_true_m > 0:
            sum_n_ij += sum_n_ii_m / sum_true_m
            num += 1

    mean_p_acc = sum_n_ij / (num + 1)
    print("The mean pixel accuracy is %f"%mean_p_acc)
    return mean_p_acc



# Input: ground truth image numpy matrix true_mat, 500*500*3.
#        predicted image numpy matrix pred_mat, 500*500*3.
#        classes list: M.
# Output: mean IOU, single value
def mean_iou(true_mat, pred_mat, M = M):
    mean_iou = 0
    x, y = true_mat.shape[0], true_mat.shape[1]
    sum_n_ij = 0
    num = 0
    for m in M:
        sum_true_m = 0
        sum_pred_m = 0
        sum_n_ii_m = 0
        for i in range(x):
            for j in range(y):
                sum_true_m += int((true_mat[i,j,:] == m).all())
                sum_pred_m += int((pred_mat[i,j,:] == m).all())
                if (true_mat[i,j,:] == m).all():
                    sum_n_ii_m += int((true_mat[i,j,:] == pred_mat[i,j,:]).all())
        if (sum_true_m + sum_pred_m - sum_n_ii_m) > 0:
            sum_n_ij += sum_n_ii_m / (sum_true_m + sum_pred_m - sum_n_ii_m)
            num += 1

    mean_iou = sum_n_ij / (num + 1)
    print("The mean iou is %f"%mean_iou)
    return mean_iou