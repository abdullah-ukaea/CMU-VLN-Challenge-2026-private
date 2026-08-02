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
    ],
    install_requires=['setuptools'],
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
            'qmapnav_node = qmapnav.mission.node:main',
        ],
    },
)
