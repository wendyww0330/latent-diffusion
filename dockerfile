# based Pythrch
FROM nvcr.io/nvidia/pytorch:23.02-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/root/.cache/torch \
    PIP_NO_CACHE_DIR=1

# dependencies + OpenCV packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgtk2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# fix pip/setuptools/wheel version
RUN python -m pip install --upgrade "pip==25.0.1" "setuptools==58.5.3" "wheel==0.38.4"

# PyTorch 2.4.1 + cu121
# torchvision / torchaudio
RUN pip install \
    "torch==2.4.1" \
    "torchvision==0.19.1+cu121" \
    "torchaudio==2.4.1+cu121" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# other packages
RUN pip install \
    "absl-py==2.3.1" \
    "aiohappyeyeballs==2.4.4" \
    "aiohttp==3.10.11" \
    "aiosignal==1.3.1" \
    "albumentations==0.4.3" \
    "altair==4.1.0" \
    "antlr4-python3-runtime==4.8" \
    "astor==0.8.1" \
    "async-timeout==5.0.1" \
    "attrs==25.3.0" \
    "backcall==0.2.0" \
    "backports.zoneinfo==0.2.1" \
    "base58==2.1.1" \
    "beautifulsoup4==4.9.3" \
    "blinker==1.8.2" \
    "brotlipy==0.7.0" \
    "cachetools==5.5.2" \
    "certifi==2020.12.5" \
    "cffi==1.14.3" \
    "chardet==3.0.4" \
    "charset-normalizer==3.4.3" \
    "click==8.1.8" \
    "cryptography==3.2.1" \
    "cycler==0.12.1" \
    "decorator==4.4.2" \
    "diffusers==0.24.0" \
    "dill==0.3.8" \
    "dnspython==2.1.0" \
    "einops==0.3.0" \
    "entrypoints==0.4" \
    "et_xmlfile==2.0.0" \
    "filelock==3.0.12" \
    "frozenlist==1.5.0" \
    "fsspec==2024.9.0" \
    "ftfy==6.1.1" \
    "future==1.0.0" \
    "gitdb==4.0.12" \
    "GitPython==3.1.30" \
    "glob2==0.7" \
    "google-auth==2.40.3" \
    "google-auth-oauthlib==1.0.0" \
    "grpcio==1.70.0" \
    "hf-xet==1.1.10" \
    "huggingface-hub==0.19.4" \
    "idna==2.10" \
    "imageio==2.35.1" \
    "imageio-ffmpeg==0.4.2" \
    "imgaug==0.2.6" \
    "importlib_metadata==8.5.0" \
    "importlib_resources==6.4.5" \
    "ipython==7.19.0" \
    "ipython-genutils==0.2.0" \
    "jedi==0.17.2" \
    "Jinja2==3.1.6" \
    "joblib==1.4.2" \
    "jsonschema==4.23.0" \
    "jsonschema-specifications==2023.12.1" \
    "kiwisolver==1.4.7" \
    "kornia==0.5.11" \
    "kornia_rs==0.1.9" \
    "lazy_loader==0.4" \
    "lpips==0.1.4" \
    "Markdown==3.7" \
    "MarkupSafe==2.1.5" \
    "matplotlib==3.3.4" \
    "mpmath==1.3.0" \
    "multidict==6.1.0" \
    "multiprocess==0.70.16" \
    "networkx==3.1" \
    "numpy==1.24.4" \
    "oauthlib==3.3.1" \
    "olefile==0.46" \
    "omegaconf==2.1.1" \
    "opencv-python-headless==4.12.0.88" \
    "openpyxl==3.1.5" \
    "packaging==25.0" \
    "pandas==1.1.5" \
    "parso==0.7.0" \
    "pexpect==4.8.0" \
    "pickleshare==0.7.5" \
    "pillow==10.2.0" \
    "pkginfo==1.7.0" \
    "pkgutil_resolve_name==1.3.10" \
    "prompt-toolkit==3.0.8" \
    "propcache==0.2.0" \
    "protobuf==5.29.5" \
    "ptyprocess==0.7.0" \
    "pudb==2019.2" \
    "pyarrow==17.0.0" \
    "pyasn1==0.6.1" \
    "pyasn1_modules==0.4.2" \
    "pycparser==2.20" \
    "pydeck==0.9.1" \
    "pyDeprecate==0.3.1" \
    "Pygments==2.7.4" \
    "pyOpenSSL==19.1.0" \
    "pyparsing==3.1.4" \
    "PySocks==1.7.1" \
    "python-dateutil==2.9.0.post0" \
    "python-etcd==0.4.5" \
    "pytz==2020.5" \
    "PyWavelets==1.4.1" \
    "PyYAML==5.3.1" \
    "referencing==0.35.1" \
    "regex==2024.11.6" \
    "requests==2.32.4" \
    "requests-oauthlib==2.0.0" \
    "rpds-py==0.20.1" \
    "rsa==4.9.1" \
    "sacremoses==0.1.1" \
    "safetensors==0.5.3" \
    "scikit-image==0.21.0" \
    "scikit-learn==1.3.2" \
    "scipy==1.10.1" \
    "six==1.15.0" \
    "smmap==5.0.2" \
    "soupsieve==2.1" \
    "streamlit==0.73.1" \
    "sympy==1.13.3" \
    "taming-transformers==0.0.1" \
    "tensorboard==2.14.0" \
    "tensorboard-data-server==0.7.2" \
    "test-tube==0.7.5" \
    "threadpoolctl==3.5.0" \
    "tifffile==2023.7.10" \
    "tokenizers==0.13.3" \
    "toml==0.10.2" \
    "toolz==1.0.0" \
    "torch-fidelity==0.3.0" \
    "torchelastic==0.2.1" \
    "torchmetrics==0.4.1" \
    "tornado==6.4.2" \
    "tqdm==4.67.1" \
    "traitlets==5.0.5" \
    "transformers==4.33.3" \
    "typing_extensions==4.13.2" \
    "tzlocal==5.2" \
    "urllib3==2.2.3" \
    "urwid==2.6.16" \
    "validators==0.34.0" \
    "watchdog==4.0.2" \
    "Werkzeug==3.0.6" \
    "xxhash==3.5.0" \
    "yarl==1.15.2" \
    "zipp==3.20.2"




# set workspace
WORKDIR /workspace

# add /workspace in PYTHONPATH
ENV PYTHONPATH=/workspace:$PYTHONPATH

# default bash
CMD ["/bin/bash"]
