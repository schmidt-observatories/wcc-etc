
# wcc-etc
ETC for the WCC

![WCC ETC Web Applet](docs/applet.png)
*Screenshot of the WCC ETC web applet interface.*

# Installation

## **1. Clone the Repository**
```sh
git clone git@github.com:schmidt-observatories/wcc-etc.git
cd wcc-etc
```


## **2. Install the Package**
Install the package itself:
```sh
pip install -e .
```

If you need to install additional dependencies, can run
```sh
pip install -r requirements.txt
```

# Tutorial
See notebooks/ directory for example tutorials.


# Running the Flask Applet

To run the web-based SNR calculator applet:

```bash
cd flask_app
python app.py
```

Then open your browser and go to:

    http://127.0.0.1:5000

You can select a config file, enter parameters, and view SNR and plots interactively.

# Notes
Configuration files (.toml) are in the config/ directory.

## Viewing Documentation Locally

To view the documentation locally, make sure you have MkDocs installed:

```bash
pip install mkdocs
```

Then, from the project root directory, run:

```bash
mkdocs serve
```

This will start a local web server. Open your browser and go to:

    http://127.0.0.1:8000

to view the documentation.
