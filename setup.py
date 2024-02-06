from setuptools import setup, find_packages
import os


def read_long_description():
    root = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(root, "README.md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


setup(
    name = 'mrl-pytorch',
    packages = find_packages(exclude=['notebooks']),
    version = '0.1.1',
    license='MIT',
    description = 'An unofficial implementation of Matryoshka Representation Learning for contrastive self-supervised learning',
    author = 'Filip Basara',
    author_email = 'basarafilip@gmail.com',
    url = 'https://github.com/filipbasara0/matryoshka-representation-learning',
    long_description=read_long_description(),
    long_description_content_type = 'text/markdown',
    keywords = [
        'machine learning',
        'pytorch',
        'self-supervised learning',
        'representation learning',
        'contrastive learning'
    ],
    install_requires=[
        'torch>=2.1',
        'torchvision>=0.16',
        'datasets>=2.15',
        'tqdm>=4.66',
        'torchinfo>=1.8.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        "console_scripts": [
            "mrl_train = run_training:main"
        ],
    },
)
