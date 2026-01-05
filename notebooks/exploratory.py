{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["# Exploratory Notebook\\n", "Prepare data and run Easy/Medium/Hard."]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["!pip -q install -r ../requirements.txt\\n", "!apt-get -qq install ffmpeg\\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["!python ../src/dataset.py --prepare\\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["!python ../src/vae.py --task easy --k 8\\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["!python ../src/vae.py --task medium --k 6\\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["!python ../src/vae.py --task hard --k 6 --beta 4.0\\n"],
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
    "language_info": { "name": "python", "version": "3.x" }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
