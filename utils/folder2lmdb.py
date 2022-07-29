import os
import os.path as osp
from PIL import Image
import six
import lmdb
import numpy as np
import pickle
import pyarrow
import torch.utils.data as data
from torch.utils.data import DataLoader
from utils.ImageFolderPaths import ImageFolderWithPaths
from utils.common import logger


def pickle_load(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)
    # return pyarrow.deserialize(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = pickle_load(txn.get(b'__len__'))
            self.keys = pickle_load(txn.get(b'__keys__'))
            self.classes = pickle_load(txn.get(b'class_name'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env

        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = pickle_load(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]
        img_name = unpacked[2]
        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target, img_name

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def pickle_dumps(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)
    # return pyarrow.serialize(obj).to_buffer()


def folder2lmdb(dpath, name="train", write_frequency=5000):
    directory = osp.expanduser(osp.join(dpath, name))
    logger.info("Loading dataset from %s" % directory)
    dataset = ImageFolderWithPaths(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=4, collate_fn=lambda x: x)
    class_name = dataset.classes

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    logger.info("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data_ in enumerate(data_loader):
        image, label, names = data_[0]

        txn.put(u'{}'.format(idx).encode('ascii'), pickle_dumps((image, label, names)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(len(data_loader))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle_dumps(keys))
        txn.put(b'__len__', pickle_dumps(len(keys)))
        txn.put(b'class_name', pickle_dumps(class_name))

    logger.info("Flushing database ...")
    db.sync()
    db.close()

