# FloodGNNs

FloodGNNs is a flood prediction project based on Graph Neural Networks (GNNs).

## Project Overview

This project aims to improve the accuracy and efficiency of flood prediction using Graph Neural Network technology. By modeling hydrological systems as graph structures, we can better capture the spatial dependencies and dynamic characteristics of river networks.

## Key Features

- GNN-based flood prediction models
- Support for multiple graph neural network architectures
- Includes benchmark models for performance comparison
- Flexible data processing and experiment configuration

## Directory Structure

- `baselines/`: Contains implementations of benchmark models
- `basicts/`: Basic tools and common components
- `experiments/`: Experiment configurations and results

## Installation

```bash
git clone https://github.com/Dreamzz5/FloodGNNs.git
cd FloodGNNs
# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Prepare the dataset
2. Configure experiment parameters
3. Run the training script
4. Evaluate model performance

### Running the Training Script

Use the following command to run the training script:

```bash
python experiments/train.py -c baselines/GNNs/config.py --g 2 -ct <model_type>
```

Where `<model_type>` can be one of the following options:
- ChebNet
- GAT
- GCNII
- GCN
- GIN
- GraphSAGE

For example, to train using the ChebNet model:

```bash
python experiments/train.py -c baselines/GNNs/config.py --g 2 -ct ChebNet
```

To use a dense adjacency matrix, add the `--d` parameter:

```bash
python experiments/train.py -c baselines/GNNs/config.py --g 2 -ct ChebNet --d
```

For detailed usage instructions, please refer to the documentation in the `docs/` directory.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
