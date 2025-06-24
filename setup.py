from setuptools import setup, find_packages

setup(
    name="chatbot_clara",
    version="1.0.1",
    description="Chatbot Clara with full notebook integration",
    author="Votre Nom",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-google-genai",
        "lime",
        "shap",
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    python_requires='>=3.7',
)
