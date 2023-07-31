import docker
from PIL import Image
import numpy as np
import cv2
import os
import shutil
from glob import glob
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()


class StyleGAN2():
    def __init__(self, verbose=True, tmp_path="./src/.tmp/stylegan2-ada-pytorch/") -> None:
        self.verbose = verbose

        self.root = os.getenv('ROOT')
        self.root_path = f"./submodules/stylegan2-ada-pytorch"
        self.tmp_path = tmp_path
        self.image_tag = "stylegan2-ada-pytorch:latest"

        self.client = docker.from_env()
        print(self.root_path)
        print(self.tmp_path)
        print("Got StyleGAN2 docker client, building image...")
        self.image, _ = self.client.images.build(path=self.root_path, tag=self.image_tag)
        print("StyleGAN2 Docker image built.")


    def project(self, target, seed=303):
        self._project_from_command([f"--target={target}", "--save-video=False", "--seed={seed}", f"--outdir={self.tmp_path}"])
        return np.load(f"{self.tmp_path}/projected_w.npz")["w"]

    def project_person(self, target_folder):
        result = []
        for img in glob(f"{target_folder}/aligned/*.jpg"):
            img_name = img.split("/")[-1].split(".")[0]
            outdir = f"{target_folder}/latents/latents_{img_name}"
            if os.path.exists(outdir):
                continue
            self._project_from_command([f"--target={img}", f"--outdir={outdir}", "--save-video=False"])
            result.append(np.load(f"{outdir}/projected_w.npz")["w"])
        return result 

    def generate_from_array(self, array):
        assert array.shape[1:] == (18, 512)
        print(f"Generating {array.shape[0]} images from array...")
        os.makedirs(self.tmp_path, exist_ok=True)
        np.savez(f'{self.tmp_path}/projected_w.npz', w=array)
        return self._generate_from_file(f'{self.tmp_path}/projected_w.npz', array.shape[0])

    def _generate_from_file(self, file: str, n: int):
        self._generate_from_command([f"--projected-w={file}"])
        return [Image.open(f"{self.tmp_path}/proj{idx:02d}.png").convert('RGB') for idx in range(n)]

    def generate_from_seed(self, seed):
        self._generate_from_command([f"--seeds={str(seed)}"])
        return Image.open(f"{self.tmp_path}/seed{seed:04d}.png").convert('RGB')

    def _run_command(self, command):
        print(command)
        container = self.client.containers.run(
            image=self.image_tag,
            auto_remove=True,
            volumes=[f"{os.getcwd()}:/scratch"],
            user=f"{os.getuid()}:{os.getgid()}",
            stderr=True,
            detach=True,
            command=command,
            shm_size="2G",
            environment=["HOME=/scratch"],
            working_dir="/scratch",
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]
        )

        if self.verbose:
            for log in container.logs(follow=True, stream=True):
                print(str(log, "utf-8"))

        result = container.wait()
        if result["StatusCode"] != 0:
            raise Exception(f"Container exited with the status code {result['StatusCode']}...")

    def _generate_from_command(self, command):
        if self.verbose: print("[INFO] StyleGAN2 - Generating image...")
        return self._run_command([
            "python3",
            f"{self.root_path}/generate.py",
            "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
            f"--outdir={self.tmp_path}",
            "--noise-mode=random"
        ] + command)

    def _project_from_command(self, command):
        if self.verbose: print("[INFO] StyleGAN2 - Projecting image...")
        return self._run_command([
            "python3",
            f"{self.root_path}/projector.py",
            "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
        ] + command)


def main():
    sg = StyleGAN2(verbose=True)
    print(sg)
    sg.generate_from_seed(1)
    sg.generate_from_seed(2)

if __name__ == "__main__":
    main()
