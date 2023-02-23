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

def add_person(person: list):
    """Write a person to the NokDB persons CSV."""
    with open(f"{root}/src/dataset/nokdb/nokdb-persons.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(person)

def add_image(image: list):
    """Write an image to the NokDB images CSV."""
    with open(f"{root}/src/dataset/nokdb/nokdb-images.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(image)

def add_sample(sample: list):
    """Write a sample to the NokDB samples CSV."""
    with open(f"{root}/src/dataset/nokdb/nokdb-samples-real.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(sample)