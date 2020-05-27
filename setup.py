from setuptools import setup, find_packages

setup(
    name='active_learning',
    version='0.1.0',
    description='',
    url='https://github.com/cjs220/active_learning_ratio_estimation',
    author='Conor Sheehan',
    author_email='conor.sheehan-2@manchester.ac.uk',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    tests_require=["pytest", "coverage"],
)
