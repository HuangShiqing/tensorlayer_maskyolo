import cv2
import matplotlib.pyplot as plt
import numpy as np


def resize_img(img):
    img_w = img.shape[1]
    img_h = img.shape[0]

    ratio = img_w / img_h
    net_w, net_h = 416, 416
    if ratio < 1:
        new_h = int(net_h)
        new_w = int(net_h * ratio)
    else:
        new_w = int(net_w)
        new_h = int(net_w / ratio)
    im_sized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    dx = net_w - new_w
    dy = net_h - new_h

    if dx > 0:
        im_sized = np.pad(im_sized, ((0, 0), (int(dx / 2), 0), (0, 0)), mode='constant', constant_values=0)
        im_sized = np.pad(im_sized, ((0, 0), (0, dx - int(dx / 2)), (0, 0)), mode='constant', constant_values=0)
    else:
        im_sized = im_sized[:, -dx:, :]
    if dy > 0:
        im_sized = np.pad(im_sized, ((int(dy / 2), 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        im_sized = np.pad(im_sized, ((0, dy - int(dy / 2)), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        im_sized = im_sized[-dy:, :, :]
    return im_sized


# def select(img, threshold=(128, 0, 0), type=(1, 0, 0)):
#     B, G, R = cv2.split(img)
#     ret, R = cv2.threshold(R, 127, 255, type=cv2.THRESH_BINARY)
#     ret, G = cv2.threshold(G, 1, 255, type=cv2.THRESH_BINARY_INV)
#     ret, G = cv2.threshold(G, 1, 255, type=cv2.THRESH_BINARY_INV)
def select(img, color=None):
    """
    img:
        shape=[416,416,3],RGB
    threshold:
        list. select the color
    new:
        shape=[416,416].
    """
    RGB = img == color
    mask = np.logical_and(RGB[:, :, 0], RGB[:, :, 1])
    mask = np.logical_and(mask, RGB[:, :, 2])
    new = np.zeros([416, 416])
    new[mask] = 255
    return new


def conv(img):
    """

    :param img:
    :return:
    """
    kernel = np.ones([8, 8]) / 255
    result = np.zeros([52, 52])
    for i in range(52):
        for j in range(52):
            result[i][j] = np.sum(np.matmul(img[i * 8:i * 8 + 8, j * 8:j * 8 + 8],
                                            kernel))  # 1 if np.sum(np.matmul(img[i * 8:i * 8 + 8, j * 8:j * 8 + 8], kernel)) > 0 else 0
    return result


def biger(max_index):
    # result = np.array(list())
    for i in range(52):
        # temp = np.array(list())
        for j in range(52):
            temp2 = np.full([8, 8], max_index[i][j])
            if j == 0:
                temp = temp2
            else:
                temp = np.hstack((temp, temp2))
        if i == 0:
            result = temp
        else:
            result = np.vstack((result, temp))
    return result


img = cv2.imread('/media/hsq/新加卷/ubuntu/data/VOC2007/VOCdevkit/VOC2007/SegmentationClass/001763.png')
origin_img = cv2.imread('/media/hsq/新加卷/ubuntu/data/VOC2007/VOCdevkit/VOC2007/JPEGImages/001763.jpg')
img = img[:, :, :: -1]
origin_img = origin_img[:, :, :: -1]
im_sized = resize_img(img)
origin_img_sized = resize_img(origin_img)

colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
          [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
          [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
          'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tvmonitor']

xml_labels = ['dog', 'cat']
results = list()
for label in labels:
    if label not in xml_labels:
        result = np.zeros([52, 52])
    else:
        color = colors[labels.index(label)]
        new = select(im_sized, color=color)
        result = conv(new)
    results.append(result)
results = np.array(results)
max_index = np.argmax(results, axis=0)

origin_mask = biger(max_index)

for i in range(3):
    for j in range(len(xml_labels)):
        origin_img_sized[:, :, i] = np.where(origin_mask == labels.index(xml_labels[j]),
                                             origin_img_sized[:, :, i] * 0.5 + 0.5 *
                                             colors[labels.index(xml_labels[j])][i],
                                             # colors[origin_mask][i]
                                             origin_img_sized[:, :, i])
plt.imshow(origin_img_sized)
plt.show()
exit()
# def apply_mask(image, mask, color, alpha=0.5):
#     """Apply the given mask to the image.
#     """
#     for c in range(3):
#         image[:, :, c] = np.where(mask == 1,
#                                   image[:, :, c] *
#                                   (1 - alpha) + alpha * color[c] * 255,
#                                   image[:, :, c])
#     return image

exit()
# plt.imshow(result, 'gray')
# plt.show()
# threshold = [128, 0, 0]
# mask = select(im_sized, threshold=threshold)
# new = np.zeros([416, 416])
# new[mask] = 255
# cv2.imwrite('new.bmp', new)
# kernel = np.ones([8, 8]) / 255
# result = np.zeros([52, 52])
# for i in range(52):
#     for j in range(52):
#         result[i][j] = np.sum(np.matmul(new[i * 8:i * 8 + 8, j * 8:j * 8 + 8], kernel))
# plt.imshow(new,'gray')
# plt.show()

# im_sized = im_sized[:, :, ::-1]
# im_sized = im_sized.copy()
# for y in range(51):
#     cv2.line(im_sized, (0, 8 * (y + 1)), (416, 8 * (y + 1)), (0, 0, 255), 1)
# for x in range(51):
#     cv2.line(im_sized, (8 * (x + 1), 0), (8 * (x + 1), 416), (0, 0, 255), 1)
# cv2.imwrite('2.bmp', im_sized)
# cv2.imwrite('1.png', im_sized, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
exit()
