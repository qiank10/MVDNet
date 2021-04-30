from setuptools import find_packages, setup

setup(
    name="mvdnet",
    version="1.0",
    author="Kun Qian",
    description="Robust Multimodal Vehicle Detection in Foggy Weather Using Complementary Lidar and Radar Signals, CVPR 2021",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=["detectron2"],
)
