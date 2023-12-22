from setuptools import setup, find_packages

setup(
    name='abcdfusion',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
    entry_points={
        'console_scripts': [
            'abcdfusion=abcdfusion.module:main',
        ],
    },
    author='Ge Shi',
    author_email='geshi@ucdavis.edu',
    description='A python package to work on ABCD dataset',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/geshijoker/ABCDFusion/tree/main',
    license='MIT',
    classifiers=[
        'Development Status :: 0.0',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
