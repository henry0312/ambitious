from setuptools import setup, find_packages


setup(
    name='Ambitious',
    version='1.0.0',
    description='some extra for Keras',
    url='https://github.com/henry0312/ambitious',
    author='Tsukasa OMOTO',
    author_email='henry0312@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],
    install_requires=['keras'],
    packages=find_packages()
)
