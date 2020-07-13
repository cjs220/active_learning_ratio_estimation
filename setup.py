from setuptools import setup

setup(
    name='active_learning_ratio_estimation',
    version='0.2.0',
    packages=['carl', 'carl.data', 'carl.ratios', 'carl.learning', 'carl.distributions',
              'active_learning_ratio_estimation',
              'active_learning_ratio_estimation.model',
              'active_learning_ratio_estimation.dataset',
              'active_learning_ratio_estimation.active_learning'],
    url='github.com/cjs220/active_learning_ratio_estimation',
    license='MIT',
    author='Conor Sheehan',
    author_email='conor.sheehan-2@manchester.ac.uk',
    description='',
    install_requires=[
        "numpy>=1.18",
        "scipy>=1.4",
        "scikit-learn>=0.23",
        "theano>=1",
        "astropy>=4",
        "six",
        "tqdm",
        "pandas>=1.0",
        "tensorflow>=2.0",
        "tensorflow-probability>=0.8"
    ]
)
