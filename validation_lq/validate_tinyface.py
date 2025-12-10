import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import tinyface_helper
import sys, os
sys.path.insert(0, os.getcwd())
from backbones import get_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from torchvision.transforms import InterpolationMode


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method='norm_weighted_avg'):
    # print(stacked_norms)
    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    if stacked_norms is not None:
        assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)
    else:
        assert fusion_method not in ['norm_weighted_avg', 'pre_norm_vector_add']

    if fusion_method == 'norm_weighted_avg':
        weights = stacked_norms / stacked_norms.sum(dim=0, keepdim=True)
        fused = (stacked_embeddings * weights).sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'pre_norm_vector_add':
        pre_norm_embeddings = stacked_embeddings * stacked_norms
        fused = pre_norm_embeddings.sum(dim=0)
        fused, fused_norm = l2_norm(fused, axis=1)
    elif fusion_method == 'average':
        fused = stacked_embeddings.sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'concat':
        fused = torch.cat([stacked_embeddings[0], stacked_embeddings[1]], dim=-1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'faceness_score':
        raise ValueError('not implemented yet. please refer to https://github.com/deepinsight/insightface/blob/5d3be6da49275602101ad122601b761e36a66a01/recognition/_evaluation_/ijb/ijb_11.py#L296')
        # note that they do not use normalization afterward.
    else:
        raise ValueError('not a correct fusion method', fusion_method)

    return fused, fused_norm


def infer(model, dataloader, use_flip_test, fusion_method, gpu):
    model.eval()
    features = []
    norms = []
    device = "cuda:" + str(gpu)
    with torch.no_grad():
        for images, idx in tqdm(dataloader):
            feature = model(images.to(device))
            if isinstance(feature, tuple):
                feature, norm = feature
            else:
                norm = torch.norm(feature, 2, 1, True)
                feature = torch.div(feature, norm)

            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images.to(device))
                if isinstance(flipped_feature, tuple):
                    flipped_feature, flipped_norm = flipped_feature
                else:
                    flipped_norm = torch.norm(flipped_feature, 2, 1, True)
                    flipped_feature = torch.div(flipped_feature, flipped_norm)

                stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                if norm is not None:
                    stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                else:
                    stacked_norms = None

                fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method=fusion_method)
                features.append(fused_feature.cpu().numpy())
                norms.append(fused_norm.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())

    features = np.concatenate(features, axis=0)
    norms = np.concatenate(norms, axis=0)
    return features, norms

def load_pretrained_model(model, model_name, gpu, num_experts, k):
    # load model and pretrained statedict
    ckpt_path = model_name
    model = get_model(model, dropout=0.0, fp16=False, num_features=512, num_experts=num_experts, k=k)

    statedict = torch.load(ckpt_path, map_location=torch.device('cuda:' + str(gpu)))
    model.load_state_dict(statedict)
    model.eval()
    return model

class ListDataset(Dataset):
    def __init__(self, img_list, image_is_saved_with_swapped_B_and_R=False, image_size=120):
        super(ListDataset, self).__init__()

        # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
        # face_emore/img has images saved with B and G (of RGB) swapped.
        # Since training data loader uses PIL (results in RGB) to read image
        # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
        # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
        # then you should set image_is_saved_with_swapped_B_and_R=True

        self.img_list = img_list
        self.transform = transforms.Compose([
            transforms.Resize(size=(120,120), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        img = cv2.imread(image_path)
        img = img[:, :, :3]
        # img = io.imread(image_path)

        if self.image_is_saved_with_swapped_B_and_R:
            # print('check if it really should be on')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx

def prepare_dataloader(img_list, batch_size, num_workers=0, image_is_saved_with_swapped_B_and_R=False, image_size=120):
    # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
    # face_emore/img has images saved with B and G (of RGB) swapped.
    # Since training data loader uses PIL (results in RGB) to read image
    # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
    # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
    # then you should set image_is_saved_with_swapped_B_and_R=True

    image_dataset = ListDataset(img_list, image_is_saved_with_swapped_B_and_R=image_is_saved_with_swapped_B_and_R, image_size=image_size)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    return dataloader


def get_save_path(model_load_path):
    directory, _ = os.path.split(model_load_path)
    results_save_path = os.path.join(directory, 'results')
    
    return results_save_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tinyface')

    parser.add_argument('--data_root', default='/cis/home/knaraya4/data')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=1024, type=int, help='')
    parser.add_argument('--model_load_path', type=str, default='/mnt/store/knaraya4/HypFace/ckpts/ms1mv3_r50/model.pt')
    parser.add_argument('--model_type', type=str, default='r50')
    parser.add_argument('--num_experts', type=int, default=2)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=120)
    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    parser.add_argument('--fusion_method', type=str, default='pre_norm_vector_add', choices=('average', 'norm_weighted_avg', 'pre_norm_vector_add', 'concat', 'faceness_score'))
    args = parser.parse_args()

    # load model
    model_load_path = args.model_load_path
    model = load_pretrained_model(args.model_type, model_load_path, args.gpu, args.num_experts, args.k)
    model.to('cuda:{}'.format(args.gpu))
    img_siz = args.image_size

    tinyface_test = tinyface_helper.TinyFaceTest(tinyface_root=args.data_root,alignment_dir_name='tinyface_aligned_112')

    # set save root
    gpu_id = args.gpu
    save_path = get_save_path(model_load_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print('save_path: {}'.format(save_path))

    img_paths = tinyface_test.image_paths
    print('total images : {}'.format(len(img_paths)))
    
    dataloader = prepare_dataloader(img_paths,  args.batch_size, num_workers=12, image_is_saved_with_swapped_B_and_R=True, image_size=img_siz)
    
    features, norms = infer(model, dataloader, use_flip_test=args.use_flip_test, fusion_method=args.fusion_method, gpu=args.gpu)
    
    results, failed_paths = tinyface_test.test_identification(features, ranks=[1,5,20], save_failures=True)
    
    print(results)
    
    # Save statistics
    pd.DataFrame({'rank':[1,5,20], 'values':results}).to_csv(os.path.join(save_path, f'tinyface_{args.fusion_method}.csv'))

    # Save failures to TXT
    failure_txt_path = os.path.join(save_path, f'tinyface_failures_rank1_{args.fusion_method}.txt')
    print(f"Saving {len(failed_paths)} failure cases to {failure_txt_path}")
    
    with open(failure_txt_path, 'w') as f:
        f.write("Index, Image_Path\n")
        for idx, path in enumerate(failed_paths):
            f.write(f"{idx}, {path}\n")