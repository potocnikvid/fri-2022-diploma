import os
import csv
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
root = os.getenv('ROOT')


def max_nokdb_pid():
    """Return the maximum NokDB PID."""
    return max([int(pid) for pid in os.listdir(f'{root}/src/dataset/nokdb') if pid.isdigit()])

def max_nokdb_iid():
    """Return the maximum NokDB IID."""
    nokdb_images_df = pd.read_csv(f"{root}/src/dataset/nokdb/nokdb-images.csv")
    return max(nokdb_images_df["iid"])

def people_names():
    """Return a list of all people names in the NokDB dataset."""
    nokdb_persons_df = pd.read_csv(f"{root}/src/dataset/nokdb/nokdb-persons.csv")
    return list(nokdb_persons_df["name"])

def get_person_iids(pid):
    """Return a list of all IIDs for a person."""
    nokdb_images_df = pd.read_csv(f"{root}/src/dataset/nokdb/nokdb-images.csv")
    return list(nokdb_images_df[nokdb_images_df["pid"] == pid]["iid"])


def add_person(person: list):
    """Write a person to the NokDB persons CSV."""
    with open(f"{root}/src/dataset/nokdb/nokdb-persons.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(person)

def add_image(image: list):
    """Write an image to the NokDB images CSV."""
    file = open(f"{root}/src/dataset/nokdb/nokdb-images.csv", "r")
    images = list(csv.reader(file, delimiter=","))[1:]
    if map(lambda i: str(i), image) not in images:
        with open(f"{root}/src/dataset/nokdb/nokdb-images.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(image)
            return image

    
def add_sample(filename, sample: list):
    """Write a sample to the NokDB samples CSV."""
    file = open(f"{root}/src/dataset/nokdb/{filename}.csv", "r")
    samples = list(csv.reader(file, delimiter=","))[1:]
    if map(lambda s: str(s), sample) not in samples:
        with open(f"{root}/src/dataset/nokdb/{filename}.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(sample)
            return sample