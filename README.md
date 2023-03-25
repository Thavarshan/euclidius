# Euclidius

Euclidius is a neural network which performs hate speech detection that when textual content is input, analyses the sentiment of the sentences and determines if the content is considered hateful, offensive or is acceptable. The out is binary meaning the text is determined to be either unacceptable (contains hate/offensive content) or acceptable.

The dataset used for this module was obtained from Kaggle and had the Creative Commons type of public license meaning the provided data is freely available for use in the public domain within the set parameters of the law of course. This dataset is from a paper on Automated Hate Speech Detection by Davidson et al, (2017).

This dataset uses data obtained from Twitter and was used to study hate speech identification. The texts are categorized as, hateful speech, language of offensive nature or acceptable language. It should be noted that due to the nature of the study this dataset contains explicit content such as vulgar language and content generally considered offensive.

### Set up

The pre-requisites to run this module on a machine is to have Python and Anaconda installed. Any version between 3.6 and 3.9 is acceptable. The module was developed and run on an Anaconda environment. The environment and all associated Python packages required for this module has been saved to an environment.yaml file and the module should be ready to run once the environment is set up correctly.

After install Python and Anaconda, to set up the environment run the following command.

```bash
cd ai && conda env create --file environment.yaml
```

This should set up the environment and install all the required packages.

Once the environment has been set up to run the API use the following command.

```bash
conda activate torchenv && flask run
```

Please make sure to run all the above given command within the ai directory.

Once the command is executed successfully the following output should be received. This indicates that the AI and its wrapper API is up and running, ready to be used.

## Contributing

Thank you for considering contributing to Euclidius! You can read the contribution guide [here](.github/CONTRIBUTING.md).

## Security Vulnerabilities

Please review [our security policy](https://github.com/Thavarshan/euclidius/security/policy) on how to report security vulnerabilities.

## License

Euclidius is open-sourced software licensed under the [MIT license](LICENSE).
