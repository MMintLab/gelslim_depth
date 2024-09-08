from setuptools import setup

setup(
    name='gelslim_depth',
    packages=['gelslim_depth'],
    version="0.0.1",
    url="https://github.com/MMintLab/gelslim_depth",
    author='WilliamvdB',
    author_email='willvdb@umich.edu',
    description="Gelslim tactile sensor depth estimation via neural networks",
    install_requires=[
        'numpy',
        'torch',
        'open3d',
        'scipy',
    ]
)