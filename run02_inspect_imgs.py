import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from skimage import io
from skimage.segmentation import find_boundaries


def segid_to_rgb(seg_id):
    r, seg_id = seg_id % 256, seg_id / 256
    g, seg_id = seg_id % 256, seg_id / 256
    b = seg_id % 256
    return np.array([r, g, b])


def rgb_to_segid(rgb):
    r, g, b = rgb
    return r + 256*g + 256*256*b


def get_boundaries(msk):
    msk_id = msk.astype(np.uint32)
    msk_id = msk[:, :, 0] + 255*msk[:, :, 1] + 255*255*msk_id[:, :, 2]
    boundaries = find_boundaries(msk_id, mode='thick')
    # img[boundaries] = 0
    return boundaries     


def plot_categories_in_legend(ann, categories):
    patches = []
    for seg in ann['segments_info']:
        color = segid_to_rgb(seg['id']) / 255.
        label = categories[seg['category_id']]['name']

        patches.append(mpatches.Patch(color=color, label=label))

    plt.subplots_adjust(right=0.7)
    plt.legend(handles=list(set(patches)),
               bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0)


def plot_img(img, msk, ann, categories):
    fig = plt.figure()

    opacity = 0.6
    res = img*opacity + msk*(1 - opacity)
    
    boundaries = get_boundaries(msk)
    res[boundaries] = 0

    plt_img = plt.imshow(res.astype(np.uint8))
    plt_img.axes.format_coord = lambda x, y: format_coord(x, y,
                                                          msk, ann, categories)
    plot_categories_in_legend(ann, categories)
    slider = add_opacity_slider(img, msk, opacity, boundaries, plt_img, fig)

    # quit by q
    plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    plt.show()


def main():
    WDIR = './__data/source'
    IDIR = os.path.join(WDIR, 'train2017')
    MDIR = os.path.join(WDIR, 'annotations', 'panoptic_train2017')
    ANN_PATH = os.path.join(WDIR, 'annotations', 'fix_panoptic_train2017.json')
    # CAT_PATH = os.path.join(WDIR, 'annotations', 'panoptic_coco_categories.json')

    with open(ANN_PATH, 'r') as f:
        dataset = json.load(f)
    # with open(CAT_PATH, 'r') as f:
    #     categories = json.load(f)

    images = dataset['images']
    annotations = dataset['annotations']
    categoriest_cnt = max(cat['id'] for cat in dataset['categories']) + 1
    categories = [{'id':-1, 'name': 'unknown'}]*categoriest_cnt
    categories[0] = {'id': 0, 'name': 'void'}

    for cat in dataset['categories']:
        categories[cat['id']] = cat

    from_idx = np.random.randint(len(images))
    to_idx = len(images)
    lst = zip(images[from_idx:to_idx], annotations[from_idx:to_idx])
    for img, ann in lst:
        # print(img['file_name'], ann['file_name'])
        img_path = os.path.join(IDIR, img['file_name'])
        msk_path = os.path.join(MDIR, ann['file_name'])

        img = io.imread(img_path)
        msk = io.imread(msk_path)

        plot_img(img, msk, ann, categories)


def add_opacity_slider(img, msk, init_opacity, boundaries, plt_img, fig):
    def update(opacity):
        res = img*opacity + msk*(1 - opacity)
        res[boundaries] = 0
        plt_img.set_array(res.astype(np.uint8))
        fig.canvas.draw_idle()

    axcolor = 'lightgoldenrodyellow'
    axopacity = plt.axes([0.15, 0.95, 0.75, 0.03], facecolor=axcolor)
    slider = Slider(axopacity, 'Opacity', 0., 1., valinit=init_opacity)
    slider.on_changed(update)

    return slider


def format_coord(x, y, msk, ann, categories):
    seg_id = rgb_to_segid(msk[int(y), int(x)])
    cat_id = 0
    if seg_id != 0:
        cat_id = [seg['category_id'] for seg in ann['segments_info'] if seg['id'] == seg_id][0]
    seg_name = categories[cat_id]['name']
    return 'x={:.4f}, y={:.4f}   {}'.format(x, y, seg_name)


def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)


if __name__ == '__main__':
    main()
