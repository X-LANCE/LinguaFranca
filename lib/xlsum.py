"""XL-Sum abstractive summarization dataset."""


import json
import os

import datasets


_CITATION = """\
@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Islam, Md. Saiful  and
      Mubasshir, Kazi  and
      Li, Yuan-Fang  and
      Kang, Yong-Bin  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.413",
    pages = "4693--4703",
}
"""


_DESCRIPTION = """\
We present XLSum, a comprehensive and diverse dataset comprising 1.35 million professionally 
annotated article-summary pairs from BBC, extracted using a set of carefully designed heuristics.
The dataset covers 45 languages ranging from low to high-resource, for many of which no
public dataset is currently available. XL-Sum is highly abstractive, concise, 
and of high quality, as indicated by human and intrinsic evaluation. 
"""

_HOMEPAGE = "https://github.com/csebuetnlp/xl-sum"

_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)"

_URL = "https://hf-mirror.com/datasets/csebuetnlp/xlsum/resolve/main/data/{}_XLSum_v{}.tar.bz2"

_LANGUAGES = [
    "oromo",
    "french",
    "amharic",
    "arabic",
    "azerbaijani",
    "bengali",
    "burmese",
    "chinese_simplified",
    "chinese_traditional",
    "welsh",
    "english",
    "kirundi",
    "gujarati",
    "hausa",
    "hindi",
    "igbo",
    "indonesian",
    "japanese",
    "korean",
    "kyrgyz",
    "marathi",
    "spanish",
    "scottish_gaelic",
    "nepali",
    "pashto",
    "persian",
    "pidgin",
    "portuguese",
    "punjabi",
    "russian",
    "serbian_cyrillic",
    "serbian_latin",
    "sinhala",
    "somali",
    "swahili",
    "tamil",
    "telugu",
    "thai",
    "tigrinya",
    "turkish",
    "ukrainian",
    "urdu",
    "uzbek",
    "vietnamese",
    "yoruba",
]


class Xlsum(datasets.GeneratorBasedBuilder):

    def __init__(self, language=None, **kwargs):
        super().__init__(
            name=f"{language}",
            description=f"Xlsum dataset for {language}",
            version=datasets.Version("2.0.0"),
            **kwargs,
        )
        self.language = language
        BUILDER_CONFIGS = [
            datasets.BuilderConfig(
                name="{}".format(self.language),
                version=datasets.Version("2.0.0")
            )
        ]
    
    VERSION = datasets.Version("2.0.0")
    

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            version=self.VERSION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        lang = str(self.config.name)
        url = _URL.format(lang, self.VERSION.version_str[:-2])

        data_dir = dl_manager.extract(url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_train.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_test.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_val.jsonl"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as f:
            for idx_, row in enumerate(f):
                data = json.loads(row)
                yield idx_, {
                    "id": data["id"],
                    "url": data["url"],
                    "title": data["title"],
                    "summary": data["summary"],
                    "text": data["text"],
                }
