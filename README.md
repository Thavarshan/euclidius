# Etherwall

Etherwall is a single project which contains two applications; one a simple forum website with a functional website and database made using PHP and MySQL and the other an AI Neural Network with an API wrapper that performs Sentiment Analysis to identify and detect hateful and or offensive language. Both applications are simple implementations and are not robust solutions for real-world usage.

The project consists of two parts; one a functioning forum discussion website where users can log in and create new discussions (threads) and to these discussion other users can make replies to; thereby propagate a conversation with each other. Text inputs are provided to type in user replies and the content is saved using a structured database that is attached to the project. The second part is the actual AI module wrapped with an API that allows access to receive and send data. The AI module is a neural network that is built using basic concepts with a manually trained model and datasets acquired through third-party means.

The idea is for when a user creates a new thread or posts a new reply the textual content before being saved into the database is sent to the AI module API and is analysed for offensive and or hateful content. If the AI model evaluates the content to be acceptable then the user’s data is saved into the database and is displayed on the forum website, otherwise an alert message is sent to the user creating the text post to notify the user of the unacceptable content present in their posts. The below diagram illustrates the flow of the idea.

## Hate speech detector API

The AI module is a neural network which performs hate speech detection that when textual content is input, analyses the sentiment of the sentences and determines if the content is considered hateful, offensive or is acceptable. The out is binary meaning the text is determined to be either unacceptable (contains hate/offensive content) or acceptable.

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

## Forum website

The forum website is built using PHP, JavaScript and MySQL. The website is functional for the most part and works just like a forum website is expected to work except for the editing and deleting parts of the posts. These were excluded on the basis of having less time to implement them. It must be noted that a web framework was used to quickly build a prototype web application without spending too much time on it.

Basic functions of the forum website include browsing list of user discussions, viewing individual discussions including replies, creating new discussions and posting replies to individual discussions.

### Set up

Before being able to run the forum website, it should be ensured that PHP and Composer is installed on the machine. If the website is going to be run on macOS, PHP and Composer can be installed via Homebrew. In addition, it is recommended to install Node and NPM.

After installing PHP and composer, run the following command inside the web directory of the project.

```bash
cd web && composer install
```

Once the above steps are complete, the project environment and application unique key must be generated.

```bash
cp .env.example .env && php artisan key:generate
```

The database does not require much attention as a simple SQLite database is used and is built into the application itself. Simply run the following command to create a SQLite database file and the database should be ready to go.

```bash
touch database/database.sqlite
```

Next is to run migrations to create the required tables and columns for the database and also to seed data into the database so that the forum website contains sample content. Do so by running the following command.

```bash
php artisan migaret:fresh --seed
```

The backend of the website is now set up and the API should be available to run, but in order for the actual website with the user interface to be accessible to the user the front-end JavaScript needs set up.

First the Node.js dependencies need to be installed. This can be accomplished by running the following command (in the same directory - web).

> just like PHP and Composer the front-end part requires Node.js installed on the machine.

```bash
npm run install
```

> Servers to run both API (forum, AI) have been built in and only require simple commands to run them.

Now that the JavaScript modules have been installed, the frontend code need to be compiled to a usable bundling. To do that, run the following command.

```bash
npm run build
```

All that is left now is to run the forum website server. Do so by running the following command.

```bash
php artisan serve
```

You may now be able to access the forum website on your browser through the address displayed on your terminal window e.g., <http://127.0.0.1:8000>.

When the given address is accessed through the browser the following screen should be visible.

You may now be able to use the following credentials to access the forum website.

| Email      | example@email.com |
| ----------- | ----------- |
| Password   | password |

## Contributing

Thank you for considering contributing to Etherwall! You can read the contribution guide [here](.github/CONTRIBUTING.md).

## Security Vulnerabilities

Please review [our security policy](https://github.com/Thavarshan/etherwall/security/policy) on how to report security vulnerabilities.

## License

Etherwall is open-sourced software licensed under the [MIT license](LICENSE).
