# Copyright 2021 Stephen Baek. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for data sets
"""

import tensorflow as tf
from pydicom import dcmread
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .tfrecords import serialize_example, decode_example
from .masks import rle2mask


def siim_to_tfrecords(siim_data_path, tfrecord_path, include_nofindings=False):
    data_path = Path(siim_data_path)
    tfrecord_path = Path(tfrecord_path)

    df_rle = pd.read_csv( str(data_path / "train-rle.csv"), index_col='ImageId' )
    dicom_paths = list(data_path.glob("dicom-images-train/*/*/*.dcm"))

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for dicom_path in tqdm(dicom_paths):
            index = dicom_path.stem

            # some images do not have rle
            try:
                rle = df_rle.loc[index].to_numpy()
            except KeyError:
                continue
            
            # read image
            dcm = dcmread(dicom_path)
            image = dcm.pixel_array

            # read masks
            mask = np.zeros(image.shape)
            for encoded in rle:
                if not isinstance(encoded, str):
                    encoded = encoded[0]
                if encoded.strip() != '-1':
                    mask += rle2mask(encoded, image.shape[1], image.shape[0])
            mask = (mask.T > 0).astype(np.uint8)*255

            # append them to the tfrecord file
            if np.sum(mask) == 0 and not include_nofindings:
                continue
            writer.write(serialize_example(image, mask))


def load_siim(data_path, include_nofindings=False):
    """Loads image and mask data from the SIIM-ACR dataset.
    Args:
        data_path: path to directory where siim dataset is downloaded.
        include_nofindings: siim dataset comes with images with no findings.
            By default, images with no findings will be excluded. To rather
            include them, set this flag to True.
    Returns:
        tf.data.Dataset object containing (image, mask) tuples.
    """
    data_path = Path(data_path)
    tfrecord_path = 'siim'
    if include_nofindings:
        tfrecord_path += '_all.tfrecords'
    else:
        tfrecord_path += '_findings_only.tfrecords'
    tfrecord_path = data_path / tfrecord_path

    if not tfrecord_path.exists():
        siim_to_tfrecords(data_path, tfrecord_path, include_nofindings)

    ds = tf.data.TFRecordDataset(str(tfrecord_path))
    return ds.map(decode_example)
