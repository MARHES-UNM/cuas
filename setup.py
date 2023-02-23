from setuptools import setup

setup(name='cuas',
      version='0.0.1',
      install_requires=["ray[rllib]",'gym>=0.18.0',
                        'pyglet>=1.5.0']
)
