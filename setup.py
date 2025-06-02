from setuptools import setup, find_packages

setup(
    name="KAN4ICH",
    version="0.1.0",
    author="Daniel K. K. Dang",
    author_email="kkdang2707.dev@gmail.com",
    description="KAN4ICH: A Python package for KAN-based and FastKAN image classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kkdang14/KAN4ICH",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",

    ],
    entry_points={
        "console_scripts": [
            "kan4ich=KAN4ICH.__main__:main",
            "fastkan4ich=KAN4ICH.fastkan.__main__:main",
            "efficientnetv2fastkan4ich=KAN4ICH.efficientnetv2fastkan.__main__:main",
            "basiccnnkan4ich=KAN4ICH.basiccnnkan.__main__:main",
            "basiccnnfastkan4ich=KAN4ICH.basiccnnfastkan.__main__:main",
            "densenetkan4ich=KAN4ICH.densenetkan.__main__:main",
            "densenetfastkan4ich=KAN4ICH.densenetfastkan.__main__:main",
            "convnextkan4ich=KAN4ICH.convnextkan.__main__:main",
            "convnextfastkan4ich=KAN4ICH.convnextfastkan.__main__:main",

        ],
    },
)