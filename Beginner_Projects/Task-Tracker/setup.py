from setuptools import setup, find_packages

setup(
    name='task-tracker',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'rich',
    ],
    entry_points={
        'console_scripts': [
            'taskr=task_tracker.cli:main',
        ],
    },
)