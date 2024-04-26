# Euclidius

Euclidius is a neural network-based application that performs hate speech detection. When textual content is input, it analyses the sentiment of the sentences and determines if the content is considered hateful, offensive, or acceptable. The output is binary, meaning the text is determined to be either unacceptable (contains hate/offensive content) or acceptable.

The model leverages advancements in natural language processing to evaluate text, using a structured approach that ensures high accuracy and performance. Below is a simple example of how you might interact with the API using Python:

```python
import requests

response = requests.post('http://localhost:5000/detect', json={"text": "example text"})
print(response.json())
```

### Dataset

The dataset used for this module was sourced from Kaggle under a Creative Commons type of public license, which allows for free use within the legal constraints. This dataset originates from a paper on Automated Hate Speech Detection by Davidson et al., (2017). The dataset comprises texts obtained from Twitter, categorized as hateful speech, offensive language, or acceptable language. Due to the nature of the study, this dataset includes explicit content, including vulgar language and generally offensive content. More details about the dataset and its usage can be found [here](https://www.kaggle.com/).

### Set Up

#### Prerequisites

To run this module, ensure that Python (versions 3.6 to 3.9) and Anaconda are installed on your machine. The project has been developed and tested in an Anaconda environment.

#### Environment Setup

Navigate to the project directory and set up the required environment with the following commands:

```bash
cd ai
conda env create --file environment.yaml
```

This will install all necessary packages as specified in `environment.yaml`.

#### Running the API

Activate the environment and start the application with:

```bash
conda activate torchenv
flask run
```

You should see the following output, indicating that the AI and its wrapper API are up and running:

```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

### Contributing

Thank you for considering contributing to Euclidius! If you're interested in helping improve the project, please read the [contribution guide](.github/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us. Here are a few areas where you can help:

- Adding new features or examples
- Improving the documentation or README
- Reporting or fixing bugs

### Security Vulnerabilities

If you discover a security vulnerability within Euclidius, please send an e-mail to the project team via [our security policy](https://github.com/Thavarshan/euclidius/security/policy). All security vulnerabilities will be promptly addressed.

### License

Euclidius is open-sourced software licensed under the [MIT license](LICENSE).

### Acknowledgments

Special thanks to the Kaggle community for providing the dataset and to Davidson et al. for their foundational work in hate speech detection.
