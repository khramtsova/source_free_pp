

import random
import os
import torchvision.transforms as transforms
from data.transform_functions import get_transform_dict


class KatyaTransform:
    def __init__(self, n_transforms, level, n_datasets=20, where_to_save="./"):
        custom_transform = ['cutout', 'auto_contrast',
                            'contrast', 'brightness',
                            'equalize', 'sharpness',
                            'solarize', 'color', 'posterize',
                            'shear_x', 'shear_y',
                            'translate_x', 'translate_y'
                            ]
        get_transform_dict()
        set_augmentation_space('fixed_custom', 30, custom_transform)
        if level == -1:
            self.rand_lvl = True
            self.level = random.randint(10, 20)
        else:
            self.rand_lvl = False
            self.level = level
        self.ops = []
        for i in range(n_datasets):
            rand_choice = random.sample(ALL_TRANSFORMS, k=n_transforms)
            self.ops.append(rand_choice)
        print(self.ops)
        # if not os.path.isdir(where_to_save):
        #    os.mkdir(where_to_save)
        # with open(where_to_save + '/setup.txt', 'w') as f:
        #    f.write(str(self.ops))
        self.n_datasets = n_datasets
        self.to_tensor = transforms.ToTensor()
        self.to_PIL = transforms.ToPILImage()

    """
    def __call__(self, img):
        # ops = random.choices(ALL_TRANSFORMS, k=1)
        # probability, level
        # img = self.ops.pil_transformer(1., self.level)(img)
        for op in self.ops:
            img = op.pil_transformer(1, self.level)(img)
        return img
    """

    def __len__(self):
        return int(self.n_datasets)

    def __call__(self, image):

        image = self.to_tensor(image)
        image = self.to_PIL(image)
        for op in self.ops[self.transf_indx]:
            # import torch
            # image.dtype = torch.float64
            image = op.pil_transformer(1, self.level)(image)
        image = self.to_tensor(image)

        return image

    def set_transform_indx(self, i):
        # Change the transform type and return the new transformations
        self.transf_indx = i
        if self.rand_lvl:
            self.level = random.randint(10, 20)
        # print("Current transform:", self.ops[self.transf_indx] )
        return str(self.ops[self.transf_indx]) + "lvl" + str(self.level)

    def apply_transform(self, i, image):
        for op in self.ops[i]:
            image = op.pil_transformer(1, self.level)(image)
        return image
