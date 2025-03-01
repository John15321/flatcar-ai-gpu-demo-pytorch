[![semantic-release](https://img.shields.io/badge/semantic--release-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

# Flatcar AI GPU Demo PyTorch (WORK IN PROGRESS)

A demo project shocasing using Flatcar with PyTorch.


# How to Use

## Installation

You can install the package using pip from a local wheel file (it can be found in the releases section):


### From a local wheel file:
```bash
pip install pip install ./dist/flatcar_ai_gpu_demo_pytorch-x.x.x-py3-none-any.whl
```

Alternatively, if you want to use the repository directly:
```bash
git clone https://github.com/yourusername/flatcar-ai-gpu-demo-pytorch.git
cd flatcar-ai-gpu-demo-pytorch
poetry install
```

# The MNIST Fashion example

## Training the Model
To train a new model on the Fashion-MNIST dataset, run the following command:

```bash
fashion-mnist-train --batch-size 64 --learning-rate 0.001 --epochs 5 --log-dir runs/fashion_mnist --model-path fashion_mnist.pth
```

This will:
- Train a neural network on the Fashion-MNIST dataset.
- Save the trained model as `fashion_mnist.pth`.
- Log training metrics to TensorBoard.

## Monitoring Training with TensorBoard (Optional)
After starting the training, you can monitor its progress using TensorBoard:

```bash
tensorboard --logdir=runs/fashion_mnist --port 6006
```

Open [http://localhost:6006](http://localhost:6006) in a browser to see the logs.

## Running Inference (Making Predictions)
To predict the class of an image, use:

```bash
fashion-mnist-predict fashion_mnist_samples/bag.png --model-path fashion_mnist.pth
```

This will output something like:

```bash
Predicted class: 8 (Bag)
```

## Downloading Fashion-MNIST Dataset Samples
To download and extract a batch of Fashion-MNIST sample images, run:

```bash
download-mnist-samples --num-samples 10 --output-dir fashion_mnist_samples
```

This will create a `fashion_mnist_samples/` directory containing a variety of example images from the dataset.

---

# CLI chatbot feature

## **Using the CLI Chatbot**

The `flatcar_ai_gpu_demo_pytorch` package includes a **command-line chatbot** powered by a locally hosted **GPT-2 model**. This chatbot allows you to have interactive conversations directly from your terminal.

### **Running the Chatbot**
To start the chatbot, after installing the package simply run the following command:

```bash
chatbot-cli
```


### **How It Works**
- The chatbot uses the **GPT-2 language model**, running locally on your machine.
- If a **GPU (CUDA)** is available, it will automatically use it for faster responses.
- The chatbot supports **top-k and top-p sampling**, making its responses more diverse and natural.

### **Example Usage**
```bash
$ chatbot-cli
You: Hello!
Chatbot: Hello! How can I assist you today?
You: What's your name?
Chatbot: I'm a language model running locally on your machine.
You: exit
Chatbot: Goodbye!
```

### **Additional Notes**
- The chatbot **does not require internet access** once the model is downloaded.
- To **improve response time**, use a GPU if available.

---

# Development

## Tools and local environment


This is a short description of how to work with this project. First create your virtual environment and activate it:

```bash
python -m venv venv
source ./venv/bin/activate
```

or

```bash
poetry shell
```

Install [Poetry](https://python-poetry.org/), its a project management tool, its used during the development to among many things build the package, install and manage dependencies. On the official website there are multiple ways of installing it but the easiest one is to simply install it in your venv with pip:

```bash
pip install poetry
```

Now you can install crucial dependencies. This command will install both package dependencies and development dependencies like `tox` (its similar to a Makefile but for Python), that will also install the package itself in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

```bash
poetry install
```

## Working with tox

---

[Tox](https://tox.wiki/en/latest/) is a generic virtual environment management and test command line tool you can use for:

- checking your package builds and installs correctly under different environments (such as different Python implementations, versions or installation dependencies),

- running your tests in each of the environments with the test tool of choice,

- acting as a frontend to continuous integration servers, greatly reducing boilerplate and merging CI and shell-based testing.

In the `tox.ini` file there a many jobs defined that perform tests, check formatting of the code, format code, lint etc. Their definition can be found but the general ones, that are also checked in the CI are:

- `lint` - Runs PyLint over the code base, both source and tests
- `lint-warn` - Runs PyLint over the code base, both source and tests but checks only for possible warnings, errors and failures, omitting style related concerns
- `check_format` - Checks formatting of the source code
- `format` - Formats the source code with `black`
- `test` - Runs all tests under `tests/`
- `type_checking` - Checks static typing of the source code using `mypy`
- `cov` - Generates and checks test coverage of the source code

Usage of any of those command is very simple, we simply run tox and specify the environment:

```bash
tox -e lint
```

If you want to run the main ones all at once simply run:

```bash
tox
```

Additionally adding the `-p` option will run the commands in parallel.

## Contributing and Releasing

---

The repository follows a semantic release development and release cycle.
This means that all PRs merged into `main`/`master` need to have formats like these:

- `feat(ABC-123): Adds /get api response`
- `fix(MINOR): Fix typo in the CI`
- `fix(#12345): Fix memory leak`
- `ci(Just about anything here): Update Python versions in the CI`

Here is the exact enforced regular expression:

```regex
'^(fix|feat|docs|test|perf|ci|chore)\([^)]+\): .+'
```

Allowed types of conventional commits:

- `fix`: a commit that fixes a bug.
- `feat`: a commit that adds new functionality.
- `docs`: a commit that adds or improves documentation.
- `test`: a commit that adds unit tests.
- `perf`: a commit that improves performance, without functional changes.
- `ci`: a commit that adds or improves the CI configuration.
- `chore`: a catch-all type for any other commits. For instance, if you're implementing a single feature and it makes sense to divide the work into multiple commits, you should mark one commit as feat and the rest as chore.

Releasing the package is done automatically when a commit is merged to `main`/`master`. A new release is created and the `CHANGELOG.md` is updated automatically.

More about the releasing mechanism:
<https://github.com/semantic-release/semantic-release>

# Credits

This package was created with Cookiecutter, and the
`John15321/cookiecutter-poetic-python` project template.

Cookiecutter: <https://github.com/audreyr/cookiecutter>

cookiecutter-poetic-python: <https://github.com/John15321/cookiecutter-poetic-python>
