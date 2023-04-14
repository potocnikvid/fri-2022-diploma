from glob import glob
import utils.nokdb as nokdb
import os
import shutil
import sys
import numpy as np
import pandas as pd
from utils.stylegan import StyleGAN2
from dotenv import load_dotenv
load_dotenv()
root = os.getenv("ROOT")

def get_max_iid(person_name):
    """Get the maximum iid for a person."""
    if os.path.exists(f"{root}/src/dataset/ppldb/people/{person_name}"):
        if not os.path.exists(f"{root}/src/dataset/ppldb/people/{person_name}/ppldb_{person_name}_1.jpg"):
            return 0
        return max([int(i.split(".")[0].split("_")[-1]) for i in glob(f"{root}/src/dataset/ppldb/people/{person_name}/ppldb_*.jpg") ]) #if (i.split(".")[-1] == "jpg" or i.split(".")[-1] == "png") and "_" in i and i.split(".")[0].split("_")[-1].isdigit()

def align_faces_all():
    """Align all faces in the PPLDB dataset."""
    for person in glob(f"{root}/src/dataset/ppldb/people/*"):
        person_name = person.split("/")[-1]
        if not os.path.exists(f"{person}/aligned"):
            for i, image in enumerate(glob(f"{person}/*")):
                os.rename(image, f"{person}/{person_name}_{i}.jpg")
            os.system(f"python src/utils/align_faces_parallel.py --root_path {person}") 

def project_all(output_path):
    """Project all faces in the PPLDB dataset."""
    stylegan = StyleGAN2(tmp_path=output_path)
    people = []
    for person in glob(f"{root}/src/dataset/ppldb/people/*"):
        if os.path.isdir(person) and os.path.exists(f"{person}/aligned"):
            people.append(stylegan.project_person(person))
    return people

def add_ppldb_nokdb_mapping(person_name, pid):
    """Add a mapping between PPLDB and NokDB."""
    with open(f"{root}/src/dataset/ppldb/nokdb-mapping.csv", "a") as f:
        f.write(f"{person_name},{pid}\n")

def get_pid(person_name):
    """Get the NokDB pid for a PPLDB person."""
    with open(f"{root}/src/dataset/ppldb/nokdb-mapping.csv", "r") as f:
        next(f)
        for line in f.readlines():
            if line.split(",")[0] == person_name:
                return int(line.split(",")[1])
    return None

def ppldb_to_nokdb():
    """Migrate the PPLDB dataset to the NokDB dataset."""
    images = []
    samples = []
    for person in glob(f"{root}/src/dataset/ppldb/people/*"):
        person_name = person.split("/")[-1]
        people_names = nokdb.people_names()
        if person_name not in people_names and os.path.exists(f"{person}/latents"):
            pid = nokdb.max_nokdb_pid() + 1
            pid
            add_ppldb_nokdb_mapping(person_name, pid)
            os.mkdir(f"./src/dataset/nokdb/{pid}")
            # person = pid,name,family_name,sex,father_pid,mother_pid,race,sex_code,race_code
            person_row = [pid, person.split("/")[-1], "", "", "", "", "", "", ""]
            nokdb.add_person(person_row)
            for img in glob(f"{person}/aligned/*"):
                img_name = img.split("/")[-1].split(".")[0]
                latent = f"{person}/latents/latents_{img_name}/projected_w.npz"
                iid = nokdb.max_nokdb_iid() + 1
                shutil.copy(img, f"./src/dataset/nokdb/{pid}/{iid}.png")
                shutil.copy(latent, f"./src/dataset/nokdb/{pid}/{iid}.npz")
                # image = iid,pid,age,emotion,emotion_code
                image_row = [iid, pid, "", "", ""]
                images.append(nokdb.add_image(image_row))
                sample_row = ["", "", "", "", pid, iid]
                samples.append(nokdb.add_sample("nodkb-samples-real-c2p", sample_row))
    return images, samples


def all_couples_to_nokdb():
    couples_df = pd.read_csv(f"{root}/src/dataset/ppldb/couples.csv")
    couples = couples_df.values.tolist()
    print(couples)
    samples = []
    for couple in couples:
        print(couple)
        cid, f_name, m_name = couple
        f_pid = get_pid(f_name)
        m_pid = get_pid(m_name)
        f_iids = nokdb.get_person_iids(f_pid)
        m_iids = nokdb.get_person_iids(m_pid)
        for f in f_iids:
            for m in m_iids:
                # sample = f_pid,f_iid,m_pid,m_iid,pid,iid
                sample_row = [f_pid, f, m_pid, m, "", ""]
                samples.append(nokdb.add_sample("nokdb-samples-real-p2c", sample_row))
    return samples

def couple_to_nokdb(cid):
    couples_df = pd.read_csv(f"{root}/src/dataset/ppldb/couples.csv")
    couples = couples_df.values.tolist()
    samples = []
    couple = filter(lambda x: x[0] == cid, couples)
    cid, f_name, m_name = next(couple)
    f_pid = get_pid(f_name)
    m_pid = get_pid(m_name)
    f_iids = nokdb.get_person_iids(f_pid)
    m_iids = nokdb.get_person_iids(m_pid)
    print(f_pid, m_pid, f_iids, m_iids)
    for f in f_iids:
        for m in m_iids:
            # sample = f_pid,f_iid,m_pid,m_iid,pid,iid
            print(f_pid, f, m_pid, m, "", "")
            sample_row = [f_pid, f, m_pid, m, "", ""]
            samples.append(nokdb.add_sample("nokdb-samples-real-p2c", sample_row))
    return samples