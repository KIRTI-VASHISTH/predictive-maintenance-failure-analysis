{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "162c412f-5291-417c-a5e5-86e0d8309dea",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "DATA_PATH = (\n",
    "    r\"C:\\docx\\Predictive_Maintenance\\data\"\n",
    "    r\"\\ai4i+2020+predictive+maintenance+dataset\\ai4i2020.csv\"\n",
    ")\n",
    "\n",
    "FEATURES = [\n",
    "    \"Type\",\n",
    "    \"Air temperature [K]\",\n",
    "    \"Process temperature [K]\",\n",
    "    \"Rotational speed [rpm]\",\n",
    "    \"Torque [Nm]\",\n",
    "    \"Tool wear [min]\"\n",
    "]\n",
    "\n",
    "TARGET = \"Machine failure\"\n",
    "\n",
    "CATEGORICAL_FEATURES = [\"Type\"]\n",
    "\n",
    "NUMERICAL_FEATURES = [\n",
    "    \"Air temperature [K]\",\n",
    "    \"Process temperature [K]\",\n",
    "    \"Rotational speed [rpm]\",\n",
    "    \"Torque [Nm]\",\n",
    "    \"Tool wear [min]\"\n",
    "]\n",
    "\n",
    "\n",
    "def load_data(path=DATA_PATH):\n",
    "    return pd.read_csv(path)\n",
    "\n",
    "\n",
    "def prepare_data(df):\n",
    "    X = df[FEATURES].copy()\n",
    "    y = df[TARGET].copy()\n",
    "\n",
    "    return train_test_split(\n",
    "        X,\n",
    "        y,\n",
    "        test_size=0.20,\n",
    "        random_state=42,\n",
    "        stratify=y\n",
    "    )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d91ec15-24e6-4f9b-a8f2-8d68f2f841ae",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
