from flask import Flask
import os

app = Flask(__name__, instance_relative_config=True)

try:
    os.makedirs(app.instance_path)
except OSError:
    pass

from app import routes