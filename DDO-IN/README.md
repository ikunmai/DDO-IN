## Installation

### Requirements

use the `requirements.txt` file to download dependencies.

### environment create

Create a new conda environment and activate it:

```bash
 conda env create -n DDOIN python=3.10
 conda activate DDOIN
```

Install the required packages:

```bash
pip install -r requirements.txt
```

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```


This will download the pre-trained models to the `models` directory.

Note: DDO-IN (ours) has no pre-trained weights since it is an instance-based method.


```bash
python src/test/test_DDOIN.py
```

