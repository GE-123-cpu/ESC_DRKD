import os
import argparse
import matplotlib
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage import morphology, measure
from scipy.ndimage import gaussian_filter
from fun import denormalization1, denormalization
from data_OCT import *
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from model import MultiProjectionLayer


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
plt.switch_backend('agg')


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--obj', type=str, default='OCT-Cirrusis')
    parser.add_argument('--data_type', type=str, default='OCT_Retouch')
    parser.add_argument('--data_path', type=str, default='D:/chenku/MBSN/OCT')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='./ablation/OCT_Cirrus/OCT-Cirrus/seed_ab1_3072/OCT-Cirrus_2024-11-11-2346_model.pth')
    parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=4708)
    parser.add_argument('--test_path', type=str, default='D:\chenku\Retouch\Image\RETOUCH-TrainingSet-Cirrus')

    args = parser.parse_args()
    args.save_dir = './' + args.data_type + '/' + args.obj + '/seed_{}/'.format(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load model and dataset
    args.input_channel = 1 if args.grayscale else 3
    encoder, _ = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    proj_layer = MultiProjectionLayer(base=64).to(device)

    checkpoint = torch.load(args.checkpoint_dir)


    proj_layer.load_state_dict(checkpoint['proj_layer'])
    decoder.load_state_dict(checkpoint['decoder'])

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_dataset = ImageDataset(args.test_normaldata_path, args.ab1, args.ab2, args.ab3)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    scores, test_imgs, recon_imgs, gt_list, gt_mask_list, img_scores = test(decoder, proj_layer, test_loader, encoder)
    scores = np.asarray(scores)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    img_ap = average_precision_score(gt_list, img_scores)
    print('image ROCAUC: %.3f' % (img_roc_auc * 100))
    print('image AP: %.3f' % (img_ap * 100))
    plt.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (args.obj, img_roc_auc))
    plt.legend(loc="lower right")

    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    pix_ap = average_precision_score(gt_mask.flatten(), scores.flatten())
    print('pixel_ROCAUC: %.3f' % (per_pixel_rocauc * 100))
    print('pixel_AP: %.3f' % (pix_ap * 100))

    save_dir = args.save_dir + '/' + f'seed_{args.seed}' + '/' + 'pictures_{:.4f}'.format(threshold)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, args.obj + '_roc_curve.png'), dpi=100)

    plot_fig1(args, test_imgs, scores, threshold, save_dir)


def test(decoder, proj_layer, test_loader, encoder):
    proj_layer.eval()
    decoder.eval()

    pixel_scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    image_scores = []
    for (data, mask, label) in tqdm(test_loader):
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        test_imgs.extend(data.cpu().numpy())
        gt_list.extend(label.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())
        with torch.no_grad():
            n, c, h, w = data.shape
            data = data.to(device)
            #data = data.expand(n, 3, h, w)
            inputs = encoder(data)
            feature = proj_layer(inputs)
            outputs = decoder(inputs[3], feature[1], feature[0])
            anomaly_map1_kd = torch.ones(1, 64, 64).to(device) - F.cosine_similarity(outputs[0], inputs[0])
            anomaly_map1_kd = anomaly_map1_kd.unsqueeze(0)
            anomaly_map1_kd = F.interpolate(anomaly_map1_kd, size=(h, w), mode='bilinear', align_corners=True)
            anomaly_map2_kd = torch.ones(1, 32, 32).to(device) - F.cosine_similarity(outputs[1], inputs[1])
            anomaly_map2_kd = anomaly_map2_kd.unsqueeze(0)
            anomaly_map2_kd = F.interpolate(anomaly_map2_kd, size=(h, w), mode='bilinear', align_corners=True)  # \
            anomaly_map3_kd = torch.ones(1, 16, 16).to(device) - F.cosine_similarity(outputs[2], inputs[2])
            anomaly_map3_kd = anomaly_map3_kd.unsqueeze(0)
            anomaly_map3_kd = F.interpolate(anomaly_map3_kd, size=(h, w), mode='bilinear', align_corners=True)  # \
            anomaly_map = (anomaly_map1_kd + anomaly_map2_kd + anomaly_map3_kd) / 3

        score = anomaly_map.squeeze(0).cpu().numpy()
        score = gaussian_filter(score, sigma=4)
        s = np.max(score)
        image_scores.append(s)
        pixel_scores.append(score)
        recon_imgs.extend(data.cpu().numpy())
    return pixel_scores, test_imgs, recon_imgs, gt_list, gt_mask_list, image_scores


def plot_fig(args, test_img, recon_imgs, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    scores = np.squeeze(scores, axis=1)
    # gts = np.squeeze(gts, axis=1)
    # gts = np.squeeze(gts, axis=1)
    for i in range(num):
        img = test_img[i]
        # img = denorm1(img)
        img = denormalization(img)
        recon_img = recon_imgs[i]
        recon_img = denormalization(recon_img)
        #print(gts[i].shape)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('Predicted mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, args.obj + '_{}_png'.format(i)), dpi=100)
        plt.close()


def plot_fig1(args, test_img, scores, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    scores = np.squeeze(scores, axis=1)
    for i in range(num):
        img1 = test_img[i]
        img = denormalization(img1)

        heat_map = scores[i] * 255

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        fig, ax_img = plt.subplots(1, 1)

        ax1 = ax_img.imshow(heat_map, cmap='jet', norm=norm)
        ax2 = ax_img.imshow(img, cmap='gray', interpolation='none')
        ax3 = ax_img.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img.axis('off')
        fig.savefig(os.path.join(save_dir, args.data_type + '_{}_png'.format(i)), dpi=100, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()