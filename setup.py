from setuptools import setup

setup(
    name="AttributeModelling",
    packages=['AttributeModelling'],
    install_requires=['pymongo', 'music21', 'sshtunnel', 'numpy', 'torch', 'tqdm', 'click', 'tensorboard-logger', 'scikit-learn', 'scipy', 'matplotlib']
)
