# Data analysis
- Document here the project: NLPmoviereviews
- Description: Use ML/DL to perform sentiment anaylsis on movie reviews
- Data Source: Tensorflow imdb reviews dataset
- Type of analysis: A comparison of NLP models for sentiment analysis

# Our webapp

https://nlp-movie-review-anffoy276a-ey.a.run.app


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for NLPmoviereviews in gitlab.com/sridhar211.
If your project is not set please add it:

- Create a new project on `gitlab.com/sridhar211/NLPmoviereviews`
- Then populate it:

```bash
##   e.g. if group is "sridhar211" and project_name is "NLPmoviereviews"
git remote add origin git@github.com:sridhar211/NLPmoviereviews.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
NLPmoviereviews-run
```

# Install

Go to `https://github.com/sridhar211/NLPmoviereviews` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:sridhar211/NLPmoviereviews.git
cd NLPmoviereviews
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
NLPmoviereviews-run
```
