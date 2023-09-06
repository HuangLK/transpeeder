from setuptools import find_packages, setup


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


setup(
    name='transpeeder',
    version='1.0.0',
    package_dir={"": "src"},
    packages=find_packages("src"),
    description='🤗 transformers and 🚀deepspeed',
    long_description='',
    long_description_content_type='text/markdown',
    license='Apache Software License 2.0',
    install_requires=fetch_requirements('requirements.txt'),
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Environment :: GPU :: NVIDIA CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Distributed Computing',
    ],
)