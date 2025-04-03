# TLFINet : Lightweight Steel Surface Defect Detection based on Texture-Guided Learning and Feature Interaction
- This research on TLFINet is valuable for steel surface defect detection. It solves problems faced by traditional and existing methods. TLFINet outperforms other lightweight algorithms in accuracy, complexity, and real-time performance.

# TODOs
- [ x ] Release training code.
- [ x ] Release inference code.
- [ ] Release pretrained models.

# Code Structure

```plaintext
project/
├── ultralytics/          # Configuration files for model parameters
│   ├── cfg/              # Utility function files
│   ├── data/             # Training and validation functions
│   ├── engine/           # Model definition files
│   ├── hub/              # Task-specific files
│   ├── models/           # Utility function files
│   ├── nn                # Package initialization file
│   ├── solutions         # Main program entry
│   ├── trackers/         # Utility function files
│   ├── utils/            # Utility function files
├── train.py              # Directory for storing datasets
├── test.py               # Source code directory
└── README.md             # Project documentation
```

# Acknowledgments
This repo is built upon [Ultralytics](https://github.com/ultralytics/ultralytics). Sincere thanks to their excellent work!
