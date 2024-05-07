from setuptools import setup, find_packages

setup(
    name='MoECache',
    version='0.1.0',
    author='Shunkang ZHANG',
    author_email='szhangcj@connect.ust.hk',
    description='A simple moe expert caching engine',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        # dependencies, e.g., 'numpy', 'requests'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)