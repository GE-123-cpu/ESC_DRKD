import argparse
import time
from tqdm import tqdm
from data_fluid import *
from fun import *
from utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log, EarlyStop
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve
from scipy.ndimage import gaussian_filter
import random
from model import MultiProjectionLayer
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='OCT_fluid anomaly detection')
    parser.add_argument('--obj', type=str, default='OCT')
    parser.add_argument('--data_type', type=str, default='OCT_fluid')
    parser.add_argument('--train_data_path', type=str, default='./Retinal_OCT/train/NORMAL')
    parser.add_argument('--test_normaldata_path', type=str, default='./Retinal_OCT/test/NORMAL')
    parser.add_argument('--ab1', type=str, default='./fluidsdukemarkeddataset-main/Dataset/images')
    parser.add_argument('--ab2', type=str, default='./fluidsdukemarkeddataset-main/Dataset/fluidmask')
    parser.add_argument('--epochs', type=int, default=80, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--validation_ratio', type=float, default=0)
    parser.add_argument('--grayscale', action='store_true', help='color or grayscale input image')
    parser.add_argument('--proj_lr', default=0.001, type=float)
    parser.add_argument('--distill_lr', default=0.005, type=float)
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    args = parser.parse_args()

    args.input_channel = 1 if args.grayscale else 3

    args.input_channel = 1 if args.grayscale else 3

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.prefix = time_file_str()
    args.save_dir = './' + args.data_type + '/' + args.obj + '/seed_{}/'.format(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'model_training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    encoder, _ = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    encoder.eval()
    proj_layer = MultiProjectionLayer(base=64).to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer_proj = torch.optim.Adam(list(proj_layer.parameters()), lr=args.proj_lr, betas=(0.5, 0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters()), lr=args.distill_lr, betas=(0.5, 0.999))

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = trainFolder(args.train_data_path)
    img_nums = len(train_dataset)
    train_num = int(img_nums * 0.1)
    val_num = img_nums - train_num


    test_dataset = test_fluidFolder(args.test_normaldata_path, args.ab1, args.ab2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, **kwargs)


    # start training
    save_name = os.path.join(args.save_dir, '{}_{}_model.pth'.format(args.obj, args.prefix))
    early_stop = EarlyStop(patience=20, save_name=save_name)
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(1, args.epochs + 1):
        train_data, _ = torch.utils.data.random_split(train_dataset, [train_num, val_num])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)
        train(proj_layer, decoder, epoch, train_loader, optimizer_proj, optimizer_distill, log, encoder)

        scores, test_imgs, recon_imgs, gt_list, gt_mask_list, img_scores = test(decoder, proj_layer, test_loader, encoder)
        scores = np.asarray(scores)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        gt_mask = np.asarray(gt_mask_list)
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        img_ap = average_precision_score(gt_list, img_scores)
        pix_ap = average_precision_score(gt_mask.flatten(), scores.flatten())
        val_loss = - (per_pixel_rocauc + img_roc_auc) * 100

        print_log(('epoch: {} image: {:.6f} image_ap: {:.6f} pixel:{:.6f} pixel_ap: {:.6f}'.format(epoch,
                                                                                                   img_roc_auc * 100,
                                                                                                   img_ap * 100,
                                                                                                   per_pixel_rocauc * 100,
                                                                                                   pix_ap * 100)), log)


        if (early_stop(val_loss, proj_layer, decoder, log)):
            break

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()


def train(proj_layer, decoder, epoch, train_loader, optimizer_proj, optimizer_distill, log, encoder):
    proj_layer.train()
    decoder.train()

    kd_losses = AverageMeter()
    rec_losses = AverageMeter()
    losses = AverageMeter()
    cos = CosineLoss()

    for data, img in tqdm(train_loader):
        optimizer_proj.zero_grad()
        optimizer_distill.zero_grad()
        n, c, h, w = data.shape
        #data = data.expand(n, 3, h, w)

        large_value = random.randint(150, 200)
        normal_value = random.randint(64, 100)
        small_value = random.randint(10, 32)
        weights = [0.2, 0.3, 0.5]
        values = [large_value, normal_value, small_value]
        t = random.choices(values, weights, k=1)[0]
        img_noise_m = noise_generate(img, t)
        data = data.to(device)
        noise = img_noise_m.to(device)
        input = encoder(data)
        out_noise = encoder(noise)
        output_noise = proj_layer(out_noise)
        out = decoder(input[3], output_noise[2], output_noise[1])

        rec_loss = cos(output_noise[0], input[0]) + cos(output_noise[1], input[1]) + cos(output_noise[2], input[2])
        kd_loss = cos(out[0], input[0]) + cos(out[1], input[1]) + cos(out[2], input[2])
        loss = rec_loss + kd_loss
        rec_losses.update(rec_loss.item(), data.size(0))
        kd_losses.update(kd_loss.item(), data.size(0))
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer_proj.step()
        optimizer_distill.step()

    print_log(('Train Epoch: {} rec_Loss: {:.6f} kd_Loss: {:.6f} all_Loss: {:.6f}'.format(epoch, rec_losses.avg, kd_losses.avg, losses.avg)), log)


def test(decoder, proj_layer, test_loader, encoder):
    proj_layer.eval()
    decoder.eval()

    pixel_scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    image_scores = []
    for (data, label, mask) in tqdm(test_loader):
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        test_imgs.extend(data.cpu().numpy())
        gt_list.extend(label.cpu().numpy().astype(int))
        gt_mask_list.extend(mask.cpu().numpy().astype(int).ravel())
        with torch.no_grad():
            n, c, h, w = data.shape
            data = data.to(device)
            #data = data.expand(n, 3, h, w)
            inputs = encoder(data)
            feature = proj_layer(inputs)

            outputs = decoder(inputs[3], feature[2], feature[1])

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



if __name__ == '__main__':
    main()
