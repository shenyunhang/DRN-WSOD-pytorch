import errno
import logging
import os
import pickle
import sys
from uuid import uuid4
from fvcore.common.file_io import PathManager

logger = logging.getLogger(__name__)


def save_object(obj, file_name, pickle_format=2):
    """Save a Python object by pickling it.

Unless specifically overridden, we want to save it in Pickle format=2 since this
will allow other Python2 executables to load the resulting Pickle. When we want
to completely remove Python2 backward-compatibility, we can bump it up to 3. We
should never use pickle.HIGHEST_PROTOCOL as far as possible if the resulting
file is manifested or used, external to the system.
    """
    file_name = os.path.abspath(file_name)
    # Avoid filesystem race conditions (particularly on network filesystems)
    # by saving to a random tmp file on the same filesystem, and then
    # atomically rename to the target filename.
    tmp_file_name = file_name + ".tmp." + uuid4().hex
    try:
        with open(tmp_file_name, "wb") as f:
            pickle.dump(obj, f, pickle_format)
            f.flush()  # make sure it's written to disk
            os.fsync(f.fileno())
        os.rename(tmp_file_name, file_name)
    finally:
        # Clean up the temp file on failure. Rather than using os.path.exists(),
        # which can be unreliable on network filesystems, attempt to delete and
        # ignore os errors.
        try:
            os.remove(tmp_file_name)
        except EnvironmentError as e:  # parent class of IOError, OSError
            if getattr(e, "errno", None) != errno.ENOENT:  # We expect ENOENT
                logger.info("Could not delete temp file %r", tmp_file_name, exc_info=True)
                # pass through since we don't want the job to crash


def _load_file(filename):
    if filename.endswith(".pkl"):
        with PathManager.open(filename, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        if "model" in data and "__author__" in data:
            # file is in Detectron2 model zoo format
            return data
        else:
            # assume file is from Caffe2 / Detectron1 model zoo
            if "blobs" in data:
                # Detection models have "blobs", but ImageNet models don't
                data = data["blobs"]
            data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
            return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

    loaded = super()._load_file(filename)  # load native pth checkpoint
    if "model" not in loaded:
        loaded = {"model": loaded}
    return loaded


in_path = sys.argv[1]
out_path = sys.argv[2]

in_pkl = _load_file(in_path)
out_pkl = dict()

print(in_pkl.keys())
print(in_pkl["model"].keys())

maps = {
    "fc6_1_w": "dilation6_conv1_w",
    "fc6_1_b": "dilation6_conv1_b",
    "fc7_1_w": "dilation6_conv2_w",
    "fc7_1_b": "dilation6_conv2_b",
    "fc6_2_w": "dilation12_conv1_w",
    "fc6_2_b": "dilation12_conv1_b",
    "fc7_2_w": "dilation12_conv2_w",
    "fc7_2_b": "dilation12_conv2_b",
    "fc6_3_w": "dilation18_conv1_w",
    "fc6_3_b": "dilation18_conv1_b",
    "fc7_3_w": "dilation18_conv2_w",
    "fc7_3_b": "dilation18_conv2_b",
    "fc6_4_w": "dilation24_conv1_w",
    "fc6_4_b": "dilation24_conv1_b",
    "fc7_4_w": "dilation24_conv2_w",
    "fc7_4_b": "dilation24_conv2_b",
}

in_pkl = in_pkl["model"]
for k in in_pkl.keys():
    new_k = k
    if k in maps.keys():
        new_k = maps[k]
    elif "conv" in k:
        plain = k[4]
        conv = k[6]
        rest = k[7:]
        new_k = "plain" + plain + "_0_conv" + conv + rest
    elif "fc" in k:
        fc = int(k[2])
        rest = k[3:]
        new_k = "fc" + str(fc - 5) + rest

    print(k, new_k)
    out_pkl[new_k] = in_pkl[k]

save_object(out_pkl, out_path)
