from setuptools import find_packages, setup

setup(
    name="Generative AI Project",
    version="0.0.0",
    author="Bappy Ahmed",
    author_email="entbappy73@gmail.com",
    packages=find_packages(),
    install_requires=[
        "flask>=2.2.0",
        "python-dotenv>=1.0.1",
        "pypdf>=3.8.0",
        "openai>=0.27.0",
        "pinecone>=2.2.2",
        "tqdm>=4.0.0",
    ]

)