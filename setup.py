from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    name='esiaf_gender_rec',
    version='0.0.1',
    description='Gender recognition component for the esiaf framework',
    url='---none---',
    author='rfeldhans',
    author_email='rfeldh@gmail.com',
    license='---none---',
    install_requires=[
        'speechpy',
        'soundfile',
        'speechemotionrecognition'
    ],
    packages=['esiaf_gender_rec']

)

setup(**setup_args)