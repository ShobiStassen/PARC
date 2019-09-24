import setuptools



#with open("requirements.txt", "r") as fh:
#    requirements = fh.read()
setuptools.setup(
    name='parc',
    version='0.34',
    packages=['parc',],
    license='MIT',
    author_email = 'shobana.venkat88@gmail.com',
    url = 'https://github.com/ShobiStassen/PARC',
    install_requires=['scipy','pandas','leidenalg','hnswlib'],
)
