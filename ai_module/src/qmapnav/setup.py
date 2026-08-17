from setuptools import find_packages, setup


PACKAGE_NAME = 'qmapnav'


setup(
    name=PACKAGE_NAME,
    version='0.1.0',
    packages=find_packages(exclude=('test',)),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            [f'resource/{PACKAGE_NAME}'],
        ),
        (f'share/{PACKAGE_NAME}', ['package.xml']),
        (f'share/{PACKAGE_NAME}/launch', ['launch/qmapnav.launch.py']),
        (
            f'share/{PACKAGE_NAME}/configs',
            ['configs/submission_v1.yaml'],
        ),
        (
            f'share/{PACKAGE_NAME}/data',
            ['data/colour_prototypes.json'],
        ),
    ],
    install_requires=['numpy', 'setuptools'],
    zip_safe=False,
    maintainer='Abdullah Saleem',
    maintainer_email='abdullahsaleem1080@gmail.com',
    description=(
        'Query-conditioned semantic mapping and navigation for the CMU VLN Challenge.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qmapnav_benchmark = qmapnav.evaluation.benchmark_runner:main',
            'qmapnav_numerical_benchmark = '
            'qmapnav.evaluation.numerical_benchmark:main',
            'qmapnav_object_benchmark = '
            'qmapnav.evaluation.object_reference_replay:main',
            'qmapnav_node = qmapnav.mission.node:main',
        ],
    },
)
