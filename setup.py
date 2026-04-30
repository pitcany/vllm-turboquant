from setuptools import find_packages, setup

setup(
    name="turboquant",
    version="0.2.1",
    description="TurboQuant: Near-optimal KV cache quantization for LLM inference (vLLM integration)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="pitcany",
    url="https://github.com/pitcany/vllm-turboquant",
    license="GPL-3.0-or-later",
    packages=find_packages(),
    package_data={"turboquant": ["codebooks/*.json"]},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "numpy",
        "scipy",
    ],
    extras_require={
        "vllm": ["vllm>=0.17,<0.19"],
        "test": ["pytest>=7"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
    ],
)
