import argparse
from pathlib import Path as Path2
import json

import torch
from path import Path
import cv2

from dataloader import DataLoaderImgFile
from eval import evaluate
from net import WordDetectorNet
from visualization import visualize_and_plot


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--weights-path', required=True, type=lambda p: Path2(p).absolute())
    parser.add_argument('--images-path', required=True, type=lambda p: Path2(p).absolute())
    parser.add_argument('--outdir', required=True, type=lambda p: Path2(p).absolute())
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    net = WordDetectorNet()
    net.load_state_dict(torch.load( str(args.weights_path) , map_location=args.device))
    net.eval()
    net.to(args.device)

    loader = DataLoaderImgFile(Path( str(args.images_path) ), net.input_size, args.device)
    res = evaluate(net, loader, max_aabbs=1000)

    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        f = loader.get_scale_factor(i)
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs]
        img = loader.get_original_img(i)
        if args.plot:
            visualize_and_plot(img, aabbs)

        img_out = Path2( args.outdir / f'image{i}' )
        img_out.mkdir(parents=True, exist_ok=True)

        # Normalise
        img_min, img_max = img.min(), img.max()
        img = ( img - img_min ) / ( img_max - img_min )
        img *= 255
        print(type(img), img.shape)
        print(img.min(), img.max())

        for ii, aabb in enumerate(aabbs):
            aabb_rounded = aabb.enlarge_to_int_grid()
            print('save:', aabb_rounded)

            nr_out = img_out / f'nr_{ii}'
            nr_out.mkdir(parents=True, exist_ok=True)

            coordinates = {
                'ymin': int(aabb_rounded.ymin),
                'ymax': int(aabb_rounded.ymax),
                'xmin': int(aabb_rounded.xmin),
                'xmax': int(aabb_rounded.xmax),
            }
            with open(nr_out / 'coords.json', 'w') as ff:
                json.dump(coordinates, ff, indent=4)

            img_cropped = img[int(aabb_rounded.ymin):int(aabb_rounded.ymax),
                              int(aabb_rounded.xmin):int(aabb_rounded.xmax)]

            cv2.imwrite(str( nr_out / 'pic.jpg' ), img_cropped)


if __name__ == '__main__':
    main()
