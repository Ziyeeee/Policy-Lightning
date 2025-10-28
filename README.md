# Policy-Lightning

**Policy-Lightning** is a **PyTorch Lightning**-based implementation of popular policy learning algorithms. It is specifically designed for embodied **multi-agent manipulation** tasks, offering clean abstractions for training, evaluation, and integration with simulators.

## 🚀 Installation

```bash
conda create -n policy-lt python=3.12
conda activate policy-lt

# Replace 'cu***' with your CUDA version, e.g., cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu***

# Install project dependencies
pip install -r requirements.txt
```

## 📁 Data Preparation

### Data Collection

Currently, we support data collection through [RoboFactory](https://github.com/MARS-EAI/RoboFactory), an automated multi-agent simulation and recording framework built on top of [ManiSkill](https://www.maniskill.ai/).

We plan to support additional simulators and datasets in the future — contributions are welcome!
To convert your existing dataset into our supported format, please refer to the [Data Conversion Guide](docs/data_convert.md).

### Demo Dataset

We provide small-scale demo datasets for quick testing and validation. You can download them from [hugging face](https://huggingface.co/datasets/Ziyeeee/Policy-Lightning/tree/main/demo).

## 🏋️ Training

**2D Diffusion Policy:**

```bash
python workspace.py --config-name=dp2 task=2a_lift_barrier
```

**3D Diffusion Policy:**

```bash
python workspace.py --config-name=dp3 task=2a_lift_barrier_3d
```

**Custom Policy:**

To integrate your own policy architecture:

1. Implement your custom policy in `./policy/`.

2. Create a corresponding configuration in `./config/`.

*Optional (for custom tasks):*

3. Place your dataset in `./data/`.

4. Add a task configuration under `./config/task/`.

```bash
python workspace --config-name=[custom_polic] task=[custom_task]
```
