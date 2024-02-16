import copy
import numpy as np
import os
import pickle
import random
import scipy
import time
import torch
import torchvision.models as models

from collections import Counter
from itertools import zip_longest
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models.feature_extraction import create_feature_extractor

class BaseDataset(Dataset):
    def __init__(self, x, y, transform=None, cls2idx=None):
        self.images = x
        self.labels = y
        self.transform = transform
        self.cls2idx = cls2idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)
        if self.cls2idx:
            y = self.cls2idx[y]
        
        return x, y

def grouper(iterable, n, fillvalue=None):
    if isinstance(n, list):
        i = 0
        group = []
        for cp in n:
            group.append(tuple(iterable[i:i+cp]))
            i += cp
        return group
    else:
        args = [iter(iterable)] * n
        group = zip_longest(*args, fillvalue=fillvalue) 
        return group        

def get_cifar100_coarse(
    train_embedding_path,
    test_embedding_path,
    val_embedding_path=None,
    validation=0.2,
    chosen_superclass=None,
    ntask=None,
    ignore_super=False
    ):
    """
    chosen_superclass: a dictionary of {superclass: [subclasses]}
    """
    

    """
    Separate the dataset generation for cifar100 20 classes
    """
    data = {}
    taskcla = []

    trainset = torch.load(train_embedding_path)
    if test_embedding_path:
        testset = torch.load(test_embedding_path)
    if val_embedding_path:
        valset = torch.load(val_embedding_path)        

    supercls, class_order = [], []
    ordered = dict(sorted(trainset["groups"].items()))
    for k, v in ordered.items():
        supercls.append(k)
        class_order.extend(v)
    
    # BRANCH FOR GIVEN CHOSEN SUPERCLASS
    if chosen_superclass:
        superclass_order, class_order = [], []
        
        for k, v in chosen_superclass.items():
            superclass_order.append(k)
            class_order.extend(sorted(v))
        
        sub2super = {v_: superclass_order.index(k) for k, v in chosen_superclass.items() for v_ in v}
        
        total_task = len(chosen_superclass) if ntask is None else ntask
        print(f"superclass_order: {superclass_order}")
        print(f"class_order: {class_order}")
        print(f"total_task: {total_task}")
        print(f"chosen_superclass: {chosen_superclass}")
            
        # for tt in range(len(chosen_superclass)):
        for tt in range(total_task):
            data[tt] = {}
            data[tt]['name'] = 'task-' + str(tt)
            data[tt]['trn'] = {'x': [], 'y': [], 'y_raw': []}
            data[tt]['val'] = {'x': [], 'y': []}
            data[tt]['tst'] = {'x': [], 'y': []}
            
        for i, (image, super_label, sub_label) in enumerate(trainset["data"]):
            # POPULATE THE TRAINSET
            # if super_label in superclass_order:
            if sub_label in class_order:
                if ntask is not None and ntask == 1:
                    data[0]['trn']['x'].append(image)
                    data[0]['trn']['y'].append(class_order.index(sub_label))                                        
                elif ignore_super:
                    data[sub2super[sub_label]]['trn']['x'].append(image)
                    data[sub2super[sub_label]]['trn']['y'].append(class_order.index(sub_label))                    
                else:
                    data[superclass_order.index(super_label)]['trn']['x'].append(image)
                    data[superclass_order.index(super_label)]['trn']['y'].append(class_order.index(sub_label))
        
        if test_embedding_path:
            # POPULATE THE TESTSET
            for i, (image, super_label, sub_label) in enumerate(testset["data"]):
                # if super_label in superclass_order:
                if sub_label in class_order:
                    if ntask is not None and ntask == 1:
                        data[0]['tst']['x'].append(image)
                        data[0]['tst']['y'].append(class_order.index(sub_label))               
                    elif ignore_super:
                        data[sub2super[sub_label]]['tst']['x'].append(image)
                        data[sub2super[sub_label]]['tst']['y'].append(class_order.index(sub_label))                                    
                    else:
                        data[superclass_order.index(super_label)]['tst']['x'].append(image)
                        data[superclass_order.index(super_label)]['tst']['y'].append(class_order.index(sub_label))
            
        # Populate validation if required
        if validation > 0.0:                            
            for tt in data.keys():
                # Divide evenly the number of classes in validation set
                total_valid = int(validation * len(data[tt]["trn"]["x"]))
                num_class = len(np.unique(data[tt]["trn"]["y"]))
                sampleperclass = np.array([total_valid // num_class] * num_class)
                
                for i in range(total_valid % num_class):
                    sampleperclass[i] += 1        
                
                for c, s in zip(np.unique(data[tt]["trn"]["y"]), sampleperclass):   
                    idx = torch.tensor(data[tt]['trn']['y']) == c
                    idx = idx.nonzero().squeeze(1).tolist()
                    val_idx = random.sample(idx, s)                
                    
                    val_x = [data[tt]['trn']['x'][i] for i in range(len(data[tt]['trn']['x'])) if i in val_idx]
                    val_y = [data[tt]['trn']['y'][i] for i in range(len(data[tt]['trn']['y'])) if i in val_idx]
                    trn_x = [data[tt]['trn']['x'][i] for i in range(len(data[tt]['trn']['x'])) if i not in val_idx]
                    trn_y = [data[tt]['trn']['y'][i] for i in range(len(data[tt]['trn']['y'])) if i not in val_idx]
                    
                    data[tt]['val']['x'].extend(val_x)
                    data[tt]['val']['y'].extend([c for _ in range(s)])
                    data[tt]['trn']['x'] = trn_x
                    data[tt]['trn']['y'] = trn_y            
                    
        # for tt in range(len(chosen_superclass)):
        for tt in range(total_task):
            data[tt]["classes"] = np.unique(data[tt]["trn"]["y"])
            data[tt]["ncla"] = len(np.unique(data[tt]["trn"]["y"]))

        n = 0
        for t in data.keys():
            taskcla.append((t, data[t]["ncla"]))
            n += data[t]["ncla"]
        data["ncla"] = n
        if ntask is not None and ntask == 1:
            data["ordered"] = {class_order.index(c): 0 for k, v in chosen_superclass.items() for c in v}            
        else:
            data["ordered"] = {class_order.index(c): superclass_order.index(k) for k, v in chosen_superclass.items() for c in v}            
    else:    
        # ORIGINAL CODE
        for tt in range(len(supercls)):
            data[tt] = {}
            data[tt]['name'] = 'task-' + str(tt)
            data[tt]['trn'] = {'x': [], 'y': []}
            data[tt]['val'] = {'x': [], 'y': []}
            data[tt]['tst'] = {'x': [], 'y': []}

        for i, (image, super_label, sub_label) in enumerate(trainset["data"]):
            # POPULATE THE TRAINSET
            data[super_label]['trn']['x'].append(image)
            data[super_label]['trn']['y'].append(class_order.index(sub_label))

        if test_embedding_path:
            # POPULATE THE TESTSET
            for i, (image, super_label, sub_label) in enumerate(testset["data"]):
                data[super_label]['tst']['x'].append(image)
                data[super_label]['tst']['y'].append(class_order.index(sub_label))

        # Populate validation if required
        if val_embedding_path:
            for i, (image, super_label, sub_label) in enumerate(valset["data"]):
                data[super_label]['val']['x'].append(image)
                data[super_label]['val']['y'].append(class_order.index(sub_label))
        elif validation > 0.0:
            for tt in data.keys():
                pop_idx = [i for i in range(len(data[tt]["trn"]["x"]))]
                val_idx = random.sample(pop_idx, int(np.round(len(pop_idx) * validation)))
                val_idx.sort(reverse=True)

                for ii in range(len(val_idx)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][val_idx[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][val_idx[ii]])
                    data[tt]['trn']['x'].pop(val_idx[ii])
                    data[tt]['trn']['y'].pop(val_idx[ii])

        for tt in range(len(supercls)):
            data[tt]["classes"] = np.unique(data[tt]["trn"]["y"])
            data[tt]["ncla"] = len(np.unique(data[tt]["trn"]["y"]))

        n = 0
        for t in data.keys():
            taskcla.append((t, data[t]["ncla"]))
            n += data[t]["ncla"]
        data["ncla"] = n
        data["ordered"] = {class_order.index(c): k for k, v in ordered.items() for c in v}
    return data, taskcla, class_order


def get_data(
    train_embedding_path, 
    test_embedding_path, 
    val_embedding_path=None,
    num_tasks=5, 
    classes_in_first_task=None, 
    validation=0.2, 
    shuffle_classes=True, 
    k=2, 
    dummy=False,
    seed=None,
    expert=False,
    superclass=False,
    class_inorder=True):
    """
    Using new data input, i.e. not raw image pixels but vision transformer's embedding, the input structure provided by
    train_embedding_path and test_embedding_path are like this:
    data = {
        'data': [
            [
                768_dimension_of_ViT_embedding,
                the_label
            ],
            [],
            ...
        ],
        'targets': labels/classes as a whole
    }    

    train_embedding_path: the path to ViT embedding train file
    test_embeding_path: the path to ViT embedding test file
    num_tasks: the number of tasks, this may be ignored if classes_in_first_task is not None
    classes_in_first_task: the number of classes in the first task. If None, the classes are divided evenly per task
    validation: floating number for validation size (e.g. 0.20)
    shuffle_classes: True/False to shuffle the class order
    k: the number of classes in the remaining tasks (only used if classes_in_first_task is not None)

    dummy: set to True to get only a small amount of data (for small testing on CPU)

    return:
    data: a dictionary of dataset for each task
        data = {
            [{
                'name': task-0,
                'train': {
                    'x': [],
                    'y': []
                },
                'val': {
                    'x': [],
                    'y': []
                },
                'test': {
                    'x': [],
                    'y': []
                },
                'classes': int
            }],
        }
    class_order: the order of the classes in the dataset (may be shuffled)
    """

    """

    """

    data = {}
    taskcla = []

    # trainset = torch_dataset(root=data_path, train=True, download=True, transform=transform)
    # testset = torch_dataset(root=data_path, train=False, download=True, transform=transform)

    trainset = torch.load(train_embedding_path)
    if test_embedding_path:
        testset = torch.load(test_embedding_path)
    if val_embedding_path:
        valset = torch.load(val_embedding_path)

    num_classes = len(np.unique(trainset["targets"]))
    # else:
    #     # this is for imagenet as there's no "targets"
    #     classes = trainset["labels"]
    #     classes = torch.stack(classes)
    #     num_classes = len(np.unique(classes))
    #     imagenet = True
    class_order = list(range(num_classes)) if class_inorder else np.unique(trainset['targets']).tolist()
    
    if shuffle_classes:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(class_order)
    print("CLASS_ORDER:", class_order)

    if classes_in_first_task is None:
        # Divide evenly the number of classes for each task
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        # Allocate the rest of the classes based on k
        remaining_classes = num_classes - classes_in_first_task
        cresttask = remaining_classes // k
        cpertask = np.array([classes_in_first_task] + [remaining_classes // cresttask] * cresttask)
        for i in range(remaining_classes % k):
            cpertask[i + 1] += 1
        num_tasks = len(cpertask)

    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    total_task = num_tasks
    for tt in range(total_task):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        # data[tt]['nclass'] = cpertask[tt]


    # Populate the train set
    # for i, (this_image, this_label) in enumerate(trainset):
    for i, (this_image, this_label) in enumerate(trainset["data"]):
        original_label = int(this_label)
        this_label = class_order.index(original_label)
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        if not expert:
            data[this_task]['trn']['y'].append(this_label - init_class[this_task])
        else:
            data[this_task]['trn']['y'].append(original_label)
        
        if dummy and i >= 500:
            break

    # Populate the test set
    # for i, (this_image, this_label) in enumerate(testset):
    if test_embedding_path:
        for i, (this_image, this_label) in enumerate(testset["data"]):
            original_label = int(this_label)
            this_label = class_order.index(original_label)
            this_task = (this_label >= cpertask_cumsum).sum()

            data[this_task]['tst']['x'].append(this_image)
            if not expert:
                data[this_task]['tst']['y'].append(this_label - init_class[this_task])
            else:
                data[this_task]['tst']['y'].append(original_label)

            if dummy and i >= 100:
                break

    # Populate validation if required
    if val_embedding_path:
        # if there's a special validation set, i.e. ImageNet
        for i, (this_image, this_label) in enumerate(valset["data"]):
            original_label = int(this_label)
            this_label = class_order.index(original_label)
            this_task = (this_label >= cpertask_cumsum).sum()

            data[this_task]['val']['x'].append(this_image)
            if not expert:
                data[this_task]['val']['y'].append(this_label - init_class[this_task])       
            else:
                data[this_task]['val']['y'].append(original_label)
    elif validation > 0.0:
        for tt in data.keys():
            pop_idx = [i for i in range(len(data[tt]["trn"]["x"]))]
            val_idx = random.sample(pop_idx, int(np.round(len(pop_idx) * validation)))
            val_idx.sort(reverse=True)

            for ii in range(len(val_idx)):
                data[tt]['val']['x'].append(data[tt]['trn']['x'][val_idx[ii]])
                data[tt]['val']['y'].append(data[tt]['trn']['y'][val_idx[ii]])
                data[tt]['trn']['x'].pop(val_idx[ii])
                data[tt]['trn']['y'].pop(val_idx[ii])     

    for tt in range(total_task):
        data[tt]["classes"] = np.unique(data[tt]["trn"]["y"])
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))

    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n    
    
    class_group = list(grouper(class_order, cpertask[0])) if classes_in_first_task is None else list(grouper(class_order, cpertask.tolist()))
    print(f"class_group: {class_group}")
    data["ordered"] = {class_order.index(c): k for k in range(len(class_group)) for c in class_group[k]}


    return data, taskcla, class_order

class Extractor:
    """
    Create a feature extractor wrapper (e.g. ViT) to generate image embedding.

    NOTE: may not be used anymore as the data input is already embedded
    """
    def __init__(self, model_, weights=None, return_nodes=None, device="cpu"):
        self.model = model_(weights=weights)
        self.weights = weights
        self.return_nodes = return_nodes
        self.feature_extractor = create_feature_extractor(self.model, return_nodes=return_nodes).to(device)
        self.feature_extractor.eval()

    def __call__(self, x):
        """
        Assuming there is only one final layer where the value needs to be extracted, therefore use index 0
        Using Vision Transformer (ViT), the last layer before the classification layer is 'getitem_5'
        """
        with torch.no_grad():
            extracted = self.feature_extractor(x)
        return extracted.get(self.return_nodes[0])

# def get_extractor(model_, weights=None, return_nodes=None):
#     model = model_(weights=weights)
#     print(f"CREATING EXTRACTOR: {return_nodes}")
#     feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

#     return feature_extractor

def extract_once(datapath, torch_dataset, transform, extractor, device, train=True, outfile=None):
    dataset = torch_dataset(root=datapath, train=train, download=True, transform=transform)
    targets = dataset.targets

    embedding = []
    for i, (image, label) in enumerate(dataset):
        image = torch.unsqueeze(image, 0).to(device)
        extracted = torch.squeeze(extractor(image)).cpu()
        embedding.append([extracted, label])
        
    extraction = {"data": embedding, "targets": targets}
    torch.save(extraction, outfile)
    
coarse_label_dict = {0: 'aquatic_mammals', 1: 'fish', 2: 'flowers',
                     3: 'food_containers', 4: 'fruit_and_vegetables', 5: 'household_electrical_devices',
                     6: 'household_furniture', 7: 'insects', 8: 'large_carnivores', 9: 'large_man-made_outdoor_things',
                     10: 'large_natural_outdoor_scenes', 11: 'large_omnivores_and_herbivores', 12: 'medium_mammals',
                     13: 'non-insect_invertebrates',14: 'people', 15: 'reptiles', 16: 'small_mammals', 
                     17: 'trees', 18: 'vehicles_1', 19: 'vehicles_2'}

fine_label_dict = {0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
                   5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle", 10: "bowl",
                   11: "boy", 12: "bridge", 13: "bus", 14: "butterfly", 15: "camel", 16: "can",
                   17: "castle", 18: "caterpillar", 19: "cattle", 20: "chair", 21: "chimpanzee",
                   22: "clock", 23: "cloud", 24: "cockroach", 25: "couch", 26: "crab", 27: "crocodile", 
                   28: "cup", 29: "dinosaur", 30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest",
                   34: "fox", 35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard", 40: "lamp",
                   41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard", 45: "lobster", 46: "man",
                   47: "maple_tree", 48: "motorcycle", 49: "mountain", 50: "mouse", 51: "mushroom", 52: "oak_tree",
                   53: "orange", 54: "orchid", 55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck",
                   59: "pine_tree", 60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
                   65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket", 70: "rose", 71: "sea",
                   72: "seal", 73: "shark", 74: "shrew", 75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 
                   79: "spider", 80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table",
                   85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor", 90: "train", 91: "trout",
                   92: "tulip", 93: "turtle", 94: "wardrobe", 95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm",}

cifar100_coarse_labels = [4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                          3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                          6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                          0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                          5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                          16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                          10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                          2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                          16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                          18,  1,  2, 15,  6,  0, 17,  8, 14, 13]

cifar100_coarse_idx = {k:v for k, v in enumerate(cifar100_coarse_labels)}    

def extract_once_coarse(datapath, torch_dataset, transform, extractor, device, train=True, outfile=None):
    coarse_label = np.array(cifar100_coarse_labels)
    dataset = torch_dataset(root=datapath, train=train, download=True, transform=transform)
    targets = coarse_label[dataset.targets].tolist()
    groups = dict()

    embedding = []
    for i, (image, label) in enumerate(dataset):
        image = torch.unsqueeze(image, 0).to(device)
        extracted = torch.squeeze(extractor(image)).cpu()
        embedding.append([extracted, cifar100_coarse_labels[label], label])

        tg = groups.get(cifar100_coarse_labels[label], [])
        tg.append(label)
        groups[cifar100_coarse_labels[label]] = tg
        
    for k, v in groups.items():
        groups[k] = np.unique(v).tolist()

    extraction = {"data": embedding, "targets": targets, "groups": groups}
    torch.save(extraction, outfile)

def extract_imagenet(path, vit_transform, extractor, device, batch_size=256, outfile=None):
    dataset = datasets.ImageFolder(path, transform=vit_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings, targets = [], []

    for i, (images, labels_) in enumerate(dataloader):
        extracted = extractor(images.to(device))
        e = extracted.detach().cpu()
        l = labels_.detach().cpu().numpy().tolist()
        embeddings.extend(zip(e, l))
        targets.extend(l)
        print(np.unique(labels_))
        torch.cuda.empty_cache()
        # break

    # NOTE: "data" should be in a pair of (embeddings, label)
    # NOTE: "labels" should not be there
    # NOTE: "targets" should store all labels in their particular order
    extraction = {"data": embeddings, "targets": targets}
    torch.save(extraction, outfile)

def get_params():
    model_ = models.vit_b_16
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    vit_transform = weights.transforms()
    return_nodes = ["getitem_5"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    return vit_transform, device, extractor

def extract_core50():    
    vit_transform, device, extractor = get_params()

    path = "core50/core50_128x128/"    
    outfile = "ds_core50_embedding.pt"

    # make static so they're in order
    # dirs = sorted(os.listdir(path))
    dirs = [f"s{i}" for i in range(1, 12)]
    embs = {}
    for dir in dirs:
        imagefolder_path = os.path.join(os.getcwd(), path, dir)
        print(imagefolder_path)
    
        dataset = datasets.ImageFolder(imagefolder_path, transform=vit_transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        embeddings, targets = [], []
        for images, labels_ in dataloader:
            extracted = extractor(images.to(device))
            e = extracted.detach().cpu()
            l = labels_.detach().cpu().numpy().tolist()
            embeddings.extend(zip(e, l))
            targets.extend(l)
            print(np.unique(labels_))
            torch.cuda.empty_cache()
        embs[dir] = {"data": embeddings, "targets": targets}
    
    torch.save(embs, outfile)

def extract_inaturalist():
    vit_transform, device, extractor = get_params()

    path = "dataset/inaturalist/train_val2018"
    outfile = "ds_inaturalist_embedding.pt"    
    dirs = ["Actinopterygii", "Amphibia", "Animalia",
            "Arachnida", "Aves", "Bacteria", "Chromista",
            "Fungi", "Insecta", "Mammalia", "Mollusca",
            "Plantae", "Protozoa", "Reptilia"]
    embs = {}
    for dir in dirs:
        imagefolder_path = os.path.join(os.getcwd(), path, dir)
        print(imagefolder_path)

        dataset = datasets.ImageFolder(imagefolder_path, transform=vit_transform)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        embeddings, targets = [], []
        for images, labels_ in dataloader:
            extracted = extractor(images.to(device))
            e = extracted.detach().cpu()
            l = labels_.detach().cpu().numpy().tolist()
            embeddings.extend(zip(e, l))
            targets.extend(l)
            print(np.unique(labels_))
            torch.cuda.empty_cache()
        embs[dir] = {"data": embeddings, "targets": targets, "classes": dataset.classes, "class_to_idx": dataset.class_to_idx}       
    torch.save(embs, outfile) 

def extract_oxflowers():
    vit_transform, device, extractor = get_params()
    path = "dataset/oxford_flowers/"
    outfile = "ds_oxford_flowers.pt"
    embs = {}
    imagefolder_path = os.path.join(os.getcwd(), path, "jpg")
    print(imagefolder_path)
    dataset = datasets.ImageFolder(imagefolder_path, transform=vit_transform)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    embeddings = []
    for images, _ in dataloader:
        extracted = extractor(images.to(device))
        e = extracted.detach().cpu()
        embeddings.extend(e)
        torch.cuda.empty_cache()
        print(images.size())

    # attach labels
    labels = scipy.io.loadmat(os.path.join(os.getcwd(), path, "imagelabels.mat"))
    labels = labels["labels"].tolist()[0]
    embeddings = [(feat, lab) for feat, lab in zip(embeddings, labels)]

    # attach setid
    setid = scipy.io.loadmat(os.path.join(os.getcwd(), path, "setid.mat"))
    trnid = setid["trnid"].tolist()[0]
    valid = setid["valid"].tolist()[0]
    tstid = setid["tstid"].tolist()[0]
    embs = {"data": embeddings, "targets": labels, "trnid": trnid, "valid": valid, "tstid": tstid}
    torch.save(embs, outfile)

def extract_mitscenes():
    vit_transform, device, extractor = get_params()
    path = "dataset/mit_scenes/"
    outfile = "mit_scenes.pt"

    imagefolder_path = os.path.join(os.getcwd(), path, "Images")
    dataset = datasets.ImageFolder(imagefolder_path, transform=vit_transform)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    embeddings, targets = [], []
    for images, labels_ in dataloader:
        extracted = extractor(images.to(device))
        e = extracted.detach().cpu()
        l = labels_.detach().cpu().numpy().tolist()
        print(np.unique(labels_))
        embeddings.extend(zip(e, l))
        targets.extend(l)
        torch.cuda.empty_cache()
    embs = {"data": embeddings, "targets": targets, "classes": dataset.classes, "class_to_idx": dataset.class_to_idx}
    torch.save(embs, outfile)

def split_mitscenes():
    mit = torch.load("mit_scenes.pt")
    path = "dataset/mit_scenes/"
    outfile = "mit_scenes.pt"
    train_all, test_all, train_names, test_names = [], [], {}, {}

    # READ TRAIN/TEST    
    with open(os.path.join(os.getcwd(), path, "TrainImages.txt")) as f:
        train_lines = f.readlines()

    with open(os.path.join(os.getcwd(), path, "TestImages.txt")) as f:
        test_lines = f.readlines()

    for line in train_lines:
        line = line.strip().split("/")
        existing_dir = train_names.get(line[0], [])
        existing_dir.append(line[1])
        train_names[line[0]] = existing_dir

    for line in test_lines:
        line = line.strip().split("/")
        existing_dir = test_names.get(line[0], [])
        existing_dir.append(line[1])
        test_names[line[0]] = existing_dir        

    for dir in mit["classes"]:
        images_path = os.path.join(os.getcwd(), path, "Images", dir)
        files = os.listdir((images_path))
        train_idx, test_idx = np.zeros(len(files)), np.zeros(len(files))

        for name in train_names[dir]:
            train_idx[files.index(name)] = 1
        for name in test_names[dir]:
            test_idx[files.index(name)] = 1

        train_all.extend(train_idx.tolist())
        test_all.extend(test_idx.tolist())
    
    train_all = (torch.tensor(train_all).int() == 1).nonzero().squeeze()
    test_all = (torch.tensor(test_all).int() == 1).nonzero().squeeze()

    print(train_all.size())
    print(test_all.size())

    features = [feat[0] for feat in mit["data"]]
    labels = [feat[1] for feat in mit["data"]]
    features = torch.stack(features)
    labels = torch.tensor(labels).int()
    print(features.size(), labels.size())

    train_features = torch.index_select(features, 0, train_all)
    test_features = torch.index_select(features, 0, test_all)
    train_labels = torch.index_select(labels, 0, train_all).cpu().tolist()
    test_labels = torch.index_select(labels, 0, test_all).cpu().tolist()

    train_features = [(f, l) for f, l in zip(train_features, train_labels)]
    test_features = [(f, l) for f, l in zip(test_features, test_labels)]

    train_ds = {"data": train_features, "targets": train_labels, "classes": mit["classes"], "class_to_idx": mit["class_to_idx"]}
    test_ds = {"data": test_features, "targets": test_labels, "classes": mit["classes"], "class_to_idx": mit["class_to_idx"]}

    torch.save(train_ds, "ds_mit_scenes_train.pt")
    torch.save(test_ds, "ds_mit_scenes_test.pt")

def separate_cub():
    path = os.path.join(os.getcwd(), "dataset/cub/CUB_200_2011")
    name_split = {}
    with open(os.path.join(path, "images.txt")) as f1, open(os.path.join(path, "train_test_split.txt")) as f2:
        images = f1.readlines()
        datasplit = f2.readlines()
        for image, split_ in zip(images, datasplit):
            _, names = image.strip().split(" ")
            _, name = names.split("/")          
            tsplit = int(split_.strip().split(" ")[1])
            name_split[name] = tsplit

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    images_path = os.path.join(path, "images")
    dirs = sorted(os.listdir(images_path))
    for dir in dirs:        
        if not os.path.exists(os.path.join(train_path, dir)):
            os.mkdir(os.path.join(train_path, dir))
        if not os.path.exists(os.path.join(test_path, dir)):
            os.mkdir(os.path.join(test_path, dir))            
        files = os.listdir(os.path.join(images_path, dir))
        for file in files:
            split_ = name_split[file]
            current = os.path.join(images_path, dir, file)
            destination = os.path.join(train_path, dir, file) if split_ == 1 else os.path.join(test_path, dir, file)
            os.rename(current, destination)                        

def separate_stanfordcars():
    path = os.path.join(os.getcwd(), "dataset/stanford_cars")
    anno = scipy.io.loadmat(os.path.join(path, "cars_annos.mat"))

    name_split = {}
    for img in anno["annotations"][0]:
        name = img[0][0].split("/")[1]
        name_split[name] = (img[5][0][0], img[6][0][0])

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    images_path = os.path.join(path, "car_ims")
    files = os.listdir(images_path)
    for file in files:
        cls_, test = name_split[file]
        if not os.path.exists(os.path.join(train_path, str(cls_))):
            os.mkdir(os.path.join(train_path, str(cls_)))
        if not os.path.exists(os.path.join(test_path, str(cls_))):
            os.mkdir(os.path.join(test_path, str(cls_)))            
        current = os.path.join(images_path, file)
        destination = os.path.join(test_path, str(cls_), file) if test == 1 else os.path.join(train_path, str(cls_), file)
        os.rename(current, destination)        

def separate_fgvcaircraft():
    path = os.path.join(os.getcwd(), "dataset/fgvc_aircraft/fgvc-aircraft-2013b/data")
    train_path = os.path.join(path, "images_variant_train.txt")
    val_path = os.path.join(path, "images_variant_val.txt")
    test_path = os.path.join(path, "images_variant_test.txt")

    with open(train_path) as tr, open(val_path) as va, open(test_path) as te:
        train_idx = [name.strip() for name in tr.readlines()]
        val_idx = [name.strip() for name in va.readlines()]
        test_idx = [name.strip() for name in te.readlines()]

    # need to replace dash -, space  and forward slash / to underscore _
    # use stupid/lazy way
    train_idx = {t.split(" ")[0]: t.split(" ")[1].replace("-", "_").replace(" ", "_").replace("/", "_") for t in train_idx}
    val_idx = {t.split(" ")[0]: t.split(" ")[1].replace("-", "_").replace(" ", "_").replace("/", "_") for t in val_idx}
    test_idx = {t.split(" ")[0]: t.split(" ")[1].replace("-", "_").replace(" ", "_").replace("/", "_") for t in test_idx}

    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")
    test_path = os.path.join(path, "test")
    images_path = os.path.join(path, "images")
    files = os.listdir(images_path)
    for file in files:
        name = file[:-4]
        if name in train_idx:
            cls_ = train_idx[name]
            if not os.path.exists(os.path.join(train_path, cls_)):
                os.mkdir(os.path.join(train_path, cls_))
            destination = os.path.join(train_path, cls_, file)
        elif name in val_idx:
            cls_ = val_idx[name]
            if not os.path.exists(os.path.join(val_path, cls_)):
                os.mkdir(os.path.join(val_path, cls_))
            destination = os.path.join(val_path, cls_, file)                
        elif name in test_idx:   
            cls_ = test_idx[name]
            if not os.path.exists(os.path.join(test_path, cls_)):
                os.mkdir(os.path.join(test_path, cls_))       
            destination = os.path.join(test_path, cls_, file)     
        current = os.path.join(images_path, file)
        print(current, destination)
        os.rename(current, destination)     

def separate_letters():
    path = os.path.join(os.getcwd(), "dataset/letters")
    train_path = os.path.join(path, "good_train.txt")
    val_path = os.path.join(path, "good_val.txt")
    test_path = os.path.join(path, "good_test.txt")

    with open(train_path) as tr, open(val_path) as va, open(test_path) as te:
        train_idx = [name.strip() for name in tr.readlines()]
        val_idx = [name.strip() for name in va.readlines()]
        test_idx = [name.strip() for name in te.readlines()]

    # English/Img/GoodImg/Bmp/Sample050/img050-00009.png
    # make a dictionary of {class: [images]} for lookup
    train_idx = letters_helper(train_idx, dict())
    val_idx = letters_helper(val_idx, dict())
    test_idx = letters_helper(test_idx, dict())

    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")
    test_path = os.path.join(path, "test")
    images_path = os.path.join(path, "English/Img/GoodImg/Bmp")
    dirs = os.listdir(images_path)

    for dir in dirs:
        if not os.path.exists(os.path.join(train_path, dir)):
            os.mkdir(os.path.join(train_path, dir))
        if not os.path.exists(os.path.join(val_path, dir)):
            os.mkdir(os.path.join(val_path, dir))
        if not os.path.exists(os.path.join(test_path, dir)):
            os.mkdir(os.path.join(test_path, dir))

        files = os.listdir(os.path.join(images_path, dir))
        for file in files:
            if file in train_idx[dir]:
                destination = os.path.join(train_path, dir, file)
            elif file in val_idx[dir]:
                destination = os.path.join(val_path, dir, file)
            elif file in test_idx[dir]:
                destination = os.path.join(test_path, dir, file)
            current = os.path.join(images_path, dir, file)
            print(current, destination)
            os.rename(current, destination)

def letters_helper(idx, dict_):
    for idx_ in idx:
        idx_ = idx_.split("/")
        cls_, name = idx_[-2], idx_[-1]
        temp = dict_.get(cls_, [])
        temp.append(name)
        dict_[cls_] = temp
    return dict_
            
def extract_generic(subdir, outfile):
    vit_transform, device, extractor = get_params()
    path = os.path.join(os.getcwd(), subdir)

    dataset = datasets.ImageFolder(path, transform=vit_transform)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    embeddings, targets = [], []
    for images, labels_ in dataloader:
        extracted = extractor(images.to(device))
        e = extracted.detach().cpu()
        l = labels_.detach().cpu().numpy().tolist()
        print(np.unique(labels_))
        embeddings.extend(zip(e, l))
        targets.extend(l)
        torch.cuda.empty_cache()
    embs = {"data": embeddings, "targets": targets, "classes": dataset.classes, "class_to_idx": dataset.class_to_idx}
    torch.save(embs, outfile)

def conv2image_svhn(path, mat_file, type):
    path = os.path.join(os.getcwd(), path)
    dir_path = os.path.join(path, type)
    
    mat = scipy.io.loadmat(os.path.join(path, mat_file))
    X = np.transpose(mat["X"], (3, 0, 1, 2))
    y = mat["y"].reshape(-1).tolist()

    for i in range(len(y)):
        if not os.path.exists(os.path.join(dir_path, str(y[i]))):
            os.mkdir(os.path.join(dir_path, str(y[i])))
        im = Image.fromarray(X[i])        
        im.save(os.path.join(dir_path, str(y[i]), f"{str(time.time_ns())}.png"))

if __name__ == "__main__":   
    # ADD SUBCLASS TARGETS FOR COARSE CIFAR100
    train_set = torch.load("cifar100_coarse_train_embedding_n.pt")
    test_set = torch.load("cifar100_coarse_test_embedding_n.pt")
    subclass_train, subclass_test = [], []

    for temp in train_set["data"]:
        subclass_train.append(temp[-1])
    extraction = {"data": train_set["data"], "targets": train_set["targets"], "groups": train_set["groups"], "subclass_targets": subclass_train}
    torch.save(extraction, "cifar100_coarse_train_embedding_nn.pt")

    for temp in test_set["data"]:
        subclass_test.append(temp[-1])
    extraction = {"data": test_set["data"], "targets": test_set["targets"], "groups": test_set["groups"], "subclass_targets": subclass_test}
    torch.save(extraction, "cifar100_coarse_test_embedding_nn.pt")        
    exit()
    ###############################################
    # transform CIFAR100
    datapath = "CIFAR_data"
    torch_dataset = datasets.CIFAR100
    model_ = models.vit_b_16
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    transform = weights.transforms()
    return_nodes = ["getitem_5"]
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    extract_once_coarse(datapath, torch_dataset, transform, extractor, device, train=True, outfile="cifar100_coarse_train_embedding_n.pt")
    extract_once_coarse(datapath, torch_dataset, transform, extractor, device, train=False, outfile="cifar100_coarse_test_embedding_n.pt")    
    exit()
    ###############################################
    # EXTRACT SVHN
    # conv2image_svhn("dataset/svhn", "train_32x32.mat", "train")
    # conv2image_svhn("dataset/svhn", "test_32x32.mat", "test")
    # extract_generic("dataset/svhn/train", "ds_svhn_train.pt")
    # extract_generic("dataset/svhn/test", "ds_svhn_test.pt")
    exit()            
    ###############################################
    # EXTRACT LETTERS
    # separate_letters()    
    # extract_generic("dataset/letters/train", "ds_letters_train.pt")
    # extract_generic("dataset/letters/val", "ds_letters_val.pt")
    # extract_generic("dataset/letters/test", "ds_letters_test.pt")
    exit()            
    ###############################################
    # EXTRACT FGVC_AIRCRAFT
    # separate_fgvcaircraft()
    # extract_generic("dataset/fgvc_aircraft/fgvc-aircraft-2013b/data/train", "ds_fgvc_aircraft_train.pt")
    # extract_generic("dataset/fgvc_aircraft/fgvc-aircraft-2013b/data/val", "ds_fgvc_aircraft_val.pt")
    # extract_generic("dataset/fgvc_aircraft/fgvc-aircraft-2013b/data/test", "ds_fgvc_aircraft_test.pt")
    exit()        
    ###############################################
    # EXTRACT iNATURALIST
    extract_inaturalist()
    exit()    
    ###############################################
    # EXTRACT CUB
    # separate_stanfordcars()
    # extract_generic("dataset/stanford_cars/train", "ds_stanford_cars_train.pt")
    # extract_generic("dataset/stanford_cars/test", "ds_stanford_cars_test.pt")
    exit()         
    ###############################################
    # EXTRACT CUB
    # separate_cub()    
    # extract_generic("dataset/cub/CUB_200_2011/train", "ds_cub_train.pt")
    # extract_generic("dataset/cub/CUB_200_2011/test", "ds_cub_test.pt")
    exit()        
    ###############################################
    # EXTRACT MIT SCENES
    # extract_mitscenes()
    # split_mitscenes()
    exit()    
    ###############################################
    # EXTRACT OXFORD FLOWERS
    extract_oxflowers()
    exit()
    ###############################################
    # EXTRACT CORE50
    extract_core50()
    exit()
    ###############################################

    train_embedding_path = "cifar100_train_embedding.pt"
    test_embedding_path = "cifar100_test_embedding.pt"
    val_embedding_path = None
    n_experts = 5

    data, class_order = get_data(
        train_embedding_path, 
        None, 
        val_embedding_path=test_embedding_path,
        num_tasks=n_experts, # num_tasks == n_experts
        expert=True
    )

    print(class_order)
    print(data.keys())
    print(data[0].keys())
    print(len(data[0]["train"]["x"]))
    print(len(data[0]["train"]["y"]))
    print(len(data[0]["val"]["x"]))
    print(len(data[0]["val"]["x"]))
    print(len(data[0]["test"]["x"]))
    print(len(data[0]["test"]["x"]))
    print(data[0]["classes"])
    exit()
    #############################################

    # inp = "imagenet100_train_embedding.pt"
    # extraction = torch.load(inp)
    # print(extraction.keys())

    # data = extraction["data"]
    # labels = extraction["labels"]

    # print(len(data))
    # print("\t", data[0].size())
    # print(len(labels))
    # print("\t", labels[-1])

    # exit()

    # print(len(emb["data"]))
    # embeddings = emb["data"][0]
    # labels = emb["data"][1]

    # print(labels)

    # exit()
    #########################################

    model_ = models.vit_b_16
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    vit_transform = weights.transforms()
    return_nodes = ["getitem_5"]
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    # path = "../Documents/imagenet-100/train/"
    # outfile = "imagenet100_train_embedding.pt"

    path = "../Documents/imagenet/val/"
    outfile = "imagenet1000_val_embedding.pt"

    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    extract_imagenet(path, vit_transform, extractor, device, batch_size=256, outfile=outfile)    

    path = "../Documents/imagenet/train/"
    outfile = "imagenet1000_train_embedding.pt"

    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    extract_imagenet(path, vit_transform, extractor, device, batch_size=256, outfile=outfile)

    exit()
    #################################################

    # torch_dataset = datasets.CIFAR10
    # datapath = "../CIFAR_data/"
    # model_ = models.vit_b_16
    # weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    # transform = weights.transforms()
    # return_nodes = ["getitem_5"]
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    # extract_once(datapath, torch_dataset, transform, extractor, device, train=True, outfile="train_embedding.pt")

    # data, class_order = get_data(
    #     torch_dataset, data_path, num_tasks=5, shuffle_classes=True, classes_in_first_task=None, k=2, validation=0.2, dummy=True)

    # for k, v in data.items():
    #     print(f"{v['name']}\n\tlen(train): {len(v['train']['x'])}\n\tlen(val): {len(v['val']['x'])}\n\tlen(test): {len(v['test']['x'])}\n\tclasses: {v['classes']}\n\tnclass: {len(v['classes'])}")

    # print(class_order)
    # cls2id = {v: k for k, v in enumerate(class_order)}
    # id2cls = {k: v for k, v in enumerate(class_order)}
    # print(cls2id)
    # print(id2cls)

    # imgarr = np.asarray(data[0]["train"]["x"][0])
    # print(data[0]["train"]["x"][0])
    # print(data[0]["train"]["y"][0])
    # print(imgarr.shape)

    ###################################################

    # transform CIFAR100
    datapath = "CIFAR_data"
    torch_dataset = datasets.CIFAR100
    model_ = models.vit_b_16
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    transform = weights.transforms()
    return_nodes = ["getitem_5"]
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    extract_once_coarse(datapath, torch_dataset, transform, extractor, device, train=True, outfile="cifar100_coarse_train_embedding.pt")
    extract_once_coarse(datapath, torch_dataset, transform, extractor, device, train=False, outfile="cifar100_coarse_test_embedding.pt")


    ###################################################

    # train_embedding_path = "cifar10_train_embedding.pt"
    # test_embedding_path = "cifar10_test_embedding.pt"

    # data, class_order = get_data(
    #     train_embedding_path, 
    #     test_embedding_path, 
    #     num_tasks=5, 
    #     validation=0.2,)

    # cls2id = {v: k for k, v in enumerate(class_order)}

    # print(class_order)
    # print(cls2id)
    # for k, v in data.items():
    #     print(f"{v['name']}\n\tlen(train): {len(v['train']['x'])}\n\tlen(val): {len(v['val']['x'])}\n\tlen(test): {len(v['test']['x'])}\n\tclasses: {v['classes']}\n\tnclass: {len(v['classes'])}")


