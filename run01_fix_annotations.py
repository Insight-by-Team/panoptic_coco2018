import json
import os
import numpy as np


def main():
    print('Sort imgs and annotations by id to match by index')
    WDIR = './__data/source/annotations'

    prefix = 'fix_'
    files = ['panoptic_train2017.json',
             'panoptic_val2017.json',
             'image_info_test2017.json']
    for f_name in files:
        print('Loading {}...'.format(f_name))
        with open(os.path.join(WDIR, f_name), 'r') as f:
            dataset = json.load(f)

        if 'images' in dataset:
            print('Sorting images...')
            dataset['images'] = sorted(dataset['images'], key=lambda obj: obj['id'])

        if 'annotations' in dataset:
            print('Sorting annotations...')
            dataset['annotations'] = sorted(dataset['annotations'],
                                            key=lambda obj: obj['image_id'])

        if 'images' in dataset and 'annotations' in dataset:
            print('Test matching: ', end='')
            print(np.all([img['id'] == ann['image_id'] for img, ann in zip(dataset['images'], dataset['annotations'])]))

        out_name = prefix + f_name
        print('Saving {}...'.format(out_name))
        with open(os.path.join(WDIR, out_name), 'w') as f:
            json.dump(dataset, f)


if __name__ == '__main__':
    main()
    
