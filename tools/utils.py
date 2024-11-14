import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import *
import cv2
import imutils
from tools.metric import *
import wandb
import csv
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from tqdm import tqdm
erodeKernSize  = 15
dilateKernSize = 11

def compute_metrics(labels, probs):
    """Validation 결과를 기반으로 메트릭 계산"""
    probs = np.array(probs)
    labels = np.array(labels)

    preds = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    precision = precision_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)

    fpr, tpr, thresholds = roc_curve(labels, probs)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return accuracy, auc, precision, f1, recall, eer


def save_results_to_csv(results, output_csv):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Id", "Probability", "Label"])
        writer.writerows(results)


def valid(dataloader, optimizer, net, device, exp_name, epoch, rank, coef):
    probs = []
    labels = []
    results = []
    running_loss = 0.0
    loss_all = 0.0
    batch_all = 0
    alpha, beta, gamma = coef[0], coef[1], coef[2]

    net.eval()
    with torch.no_grad():
        for data_idx, noise, label in tqdm(dataloader, desc=f"Validation Epoch {epoch+1}, Rank {rank}"):
            noise = noise.float().to(device)
            label = label.float().to(device)
            optimizer.zero_grad()

            # re_noise, prob, kld = net(noise)
            prob = net(noise)
            noise = noise.view(noise.size(0), -1)
            label = label.view(-1, 1)

            # localization_loss = nn.MSELoss()(re_noise, noise)
            score_loss = nn.BCEWithLogitsLoss()(prob, label)
            # elbo = alpha * -localization_loss - beta * kld - gamma * score_loss
            # loss = -elbo
            loss = score_loss
            loss_all += loss.item()
            batch_all += 1
            running_loss += loss.item()

            probs.extend(prob.cpu().numpy())
            labels.extend(label.cpu().numpy())
            for i in range(len(prob)):
                results.append([data_idx[i], torch.sigmoid(prob[i]).item(), label[i].item()])

    mean_loss = loss_all / batch_all

    # Save results for this rank
    save_results_to_csv(results, f"results/{exp_name}/val_epoch{epoch+1}_rank{rank}.csv")

    # Only rank 0 computes and returns metrics
    if rank == 0:
        accuracy, auc, precision, f1, recall, eer = compute_metrics(labels, probs)
        metrics = {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "f1": f1,
            "recall": recall,
            "eer": eer,
        }
        return mean_loss, metrics
    else:
        return mean_loss, None


def train(dataloader, optimizer, net, device, epoch, rank, coef):
    running_loss = 0.0
    loss_all = 0.0
    batch_all = 0
    alpha, beta, gamma = coef[0], coef[1], coef[2]
    net.train()
    for data_idx, noise, label in tqdm(dataloader, desc=f"Training Epoch {epoch+1}, Rank {rank}"):
        noise = noise.float().to(device)
        label = label.float().to(device)
        batchsz = noise.size(0)
        if batchsz == 1:
            continue

        optimizer.zero_grad()
        # re_noise, prob, kld = net(noise)
        prob = net(noise)
        noise = noise.view(noise.size(0), -1)
        label = label.view(-1, 1)

        # localization_loss = nn.MSELoss()(re_noise, noise)
        score_loss = nn.BCEWithLogitsLoss()(prob, label)
        # elbo = alpha * -localization_loss - beta * kld - gamma * score_loss
        # loss = -elbo
        loss = score_loss
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        batch_all += 1
        running_loss += loss.item()

    mean_loss = loss_all / batch_all
    return mean_loss
