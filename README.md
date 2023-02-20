# [Master's thesis] Predicting child appearance from facial images of parents using deep learning

Author: Matjaž Mav

Mentors: doc. dr. Luka Šajn and izr. prof. dr. Vitomir Štruc

Organization: University of Ljubljana, Faculty of Computer and Information Science

---

## Abstract
TODO


## Requirements
* Docker
* [NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-container-runtime)
* Conda (see [./conda_env.yml](./conda_env.yml))
* NVIDIA GPU, NVIDIA drivers (see [NVlabs/stylegan2-ada](https://github.com/NVlabs/stylegan2-ada))
* Create copy of the `.env` file, name it `dev.env` and fill the values
* conda create -n diploma_env python=3.9.7 deepface docker-py gitpython ipykernel ipython jupyter_client jupyter_core keras keras-preprocessing matplotlib networkx notebook numpy oauthlib opencv pandas pillow pygithub py-opencv python-dotenv pytorch pytorch-lightning requests scikit-learn scipy sklearn-contrib-lightning tensorboard tensorflow torchvision torchmetrics wandb -c conda-forge


## Notes

List of useful commands:
```sh
# List processes using NVIDIA GPU resources
$ sudo fuser -v /dev/nvidia*

# List NVIDIA GPU information
$ nvidia-smi

# Run python script detached
$ rm nohup.out && nohup python script.py &
$ tail -f nohup.out
```
- 