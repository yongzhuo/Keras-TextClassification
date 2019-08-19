# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/13 10:17
# @author   :Mo
# @function :setup of Keras-TextClassification
# @codes    :copy reference from https://github.com/TianWenQAQ/Kashgari/blob/master/setup.py

from setuptools import find_packages, setup
import pathlib
import codecs

# Package meta-data.
NAME = 'Keras-TextClassification'
DESCRIPTION = 'chinese textclassification of keras'
URL = 'https://github.com/yongzhuo/Keras-TextClassification'
EMAIL = '1903865025@qq.com'
AUTHOR = 'yongzhuo'
LICENSE = 'MIT'

with codecs.open('README.md', 'r', 'utf8') as reader:
    long_description = reader.read()
with codecs.open('requirements.txt', 'r', 'utf8') as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))

setup(name=NAME,
        version='0.1.3',
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        packages=find_packages(exclude=('test')),
        install_requires=install_requires,
        include_package_data=True,
        license=LICENSE,
        classifiers=['License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Programming Language :: Python :: Implementation :: PyPy'],)


if __name__ == "__main__":
    print("setup ok!")

# 说明，项目工程目录这里nlp_xiaojiang，实际上，下边还要有一层nlp_xiangjiang，也就是说，nlp_xiangjiang和setup同一层
# Data包里必须要有__init__.py，否则文件不会生成

# step:
#     打开cmd
#     到达安装目录
#     python setup.py build
#     python setup.py install

# python setup.py bdist_wheel --universal
#
















# pip uninstall Keras-TextClassification
