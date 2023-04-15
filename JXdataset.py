# coding=utf-8

import json
import os
import glob

import datasets

from PIL import Image
import re

import unicodedata

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
Generated data for Junction Hackathon
"""
train_file_path = "/home/viethd/workspace/junction/data/train/"
eval_file_path = "/home/viethd/workspace/junction/data/validation/"
LABEL_LIST = [p.split('.')[0] for p in os.listdir(train_file_path)]

def refine_string(s: str) -> str:
    s = re.sub("\b", " ", s)
    s = re.sub("\s+", " ", s)
    s = unicodedata.normalize("NFC", s).strip()
    return s


class JunctionConfig(datasets.BuilderConfig):
    """BuilderConfig for Junction"""

    def __init__(self, **kwargs):
        """BuilderConfig for Junction.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(JunctionConfig, self).__init__(**kwargs)


class Junction(datasets.GeneratorBasedBuilder):
    """ReJO dataset."""

    BUILDER_CONFIGS = [
        JunctionConfig(
            name="Junction",
            version=datasets.Version("0.1.0"),
            description="Junction dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=LABEL_LIST)
                    ,
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_file_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": eval_file_path},
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        for label_file in sorted(os.listdir(filepath)):
            with open(os.path.join(filepath, label_file), "r") as f:
                lines = f.readlines()
            label = os.path.splitext(label_file)[0]
            for i, sample in enumerate(lines):
                yield f"{label}_{i}", {
                    "text": refine_string(sample.lower()),
                    "label": label,
                }
