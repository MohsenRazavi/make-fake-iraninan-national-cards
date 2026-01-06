# Make Fake Iranian National Card

A python scripts which uses a sample national card and pastes a string of 10 random digits on national code.

## 🛠 Prerequisites

Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run
Run following command causes to create samples.
```bash
python main.py
```
### Configs
You can config script through changing following variables in main.py or in local_config.py (You should create the file yourself.)

Set number of sample you need in NUMBER_OF_SAMPLES.

Set MAKE_REALISTIC to True to apply realistic filters (such as blur and rotation) on samples.

