from setuptools import find_packages, setup
from glob import glob
package_name = 'rl_real_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/rl_real_py/configs', glob('configs/*.yaml')),
        ('share/rl_real_py/policy', glob('policy/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='woan',
    maintainer_email='woan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "rl_real = rl_real_py.rl_real:main",
            "rl_real_gym = rl_real_py.rl_real_gym:main",
            "rl_obs_test = rl_real_py.rl_obs_pub_test:main",
            "rl_real_thread = rl_real_py.rl_real_thread:main",
            "rl_real_xbox = rl_real_py.rl_real_xbox:main",
        ],
    },
)
