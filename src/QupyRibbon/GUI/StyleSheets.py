__author__ = 'magnus'

import os

stylesheet_instance = None


def get_stylesheet(name):
    global stylesheet_instance
    if not stylesheet_instance:
        stylesheet_instance = Stylesheets()
    return stylesheet_instance.get_stylesheet(name)


class Stylesheets(object):
    def __init__(self):
        self._stylesheets = {}
        # Get the directory where this file is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        stylesheets_dir = os.path.join(base_dir, "..", "stylesheets")

        self.make_stylesheet("main", os.path.join(stylesheets_dir, "main.css"))
        self.make_stylesheet("ribbon", os.path.join(stylesheets_dir, "ribbon.css"))
        self.make_stylesheet("ribbonPane", os.path.join(stylesheets_dir, "ribbonPane.css"))
        self.make_stylesheet("ribbonButton", os.path.join(stylesheets_dir, "ribbonButton.css"))
        self.make_stylesheet("ribbonSmallButton", os.path.join(stylesheets_dir, "ribbonSmallButton.css"))

    def make_stylesheet(self, name, path):
        with open(path) as data_file:
            stylesheet = data_file.read()

        self._stylesheets[name] = stylesheet

    def get_stylesheet(self, name):
        stylesheet = ""
        try:
            stylesheet = self._stylesheets[name]
        except KeyError:
            print("stylesheet " + name + " not found")
        return stylesheet
