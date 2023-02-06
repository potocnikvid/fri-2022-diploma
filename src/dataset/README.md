# Next of Kin Dataset

## Dataset Characteristics

Dataset contains over 120k image kin triplet combinations, all of the images have also beed converted to the StyleGAN2 extended latent W space of dimension `[1, 18, 512]`.

| Characteristic                        | Value               |
|---------------------------------------|--------------------:| 
| # persons                             | 553                 |
| # images                              | 5,017               |
| # images per person                   | 6.22 (4.30)         |
| # image sample combinations           | 127,719             |
| # father/mother/son person triplets   | 116                 |
| # father/mother/son person triplets   | 133                 |
| image resolution                      | 512x512             |

## Metadata
| Path                                 | Description                                         |
|--------------------------------------|-----------------------------------------------------|
| `├── nokdb-images.csv `              | List of images with attributes                      |
| `├── nokdb-persons.csv`              | List of persons with attributes                     |
| `├── nokdb-samples-test.csv`         | List of triplets test (father, mother, child)       |
| `├── nokdb-samples-train.csv`        | List of triplets train (father, mother, child)      |
| `├── nokdb-samples-validation.csv`   | List of triplets validation (father, mother, child) |
| `├── nokdb-normalization.npz`        | W vector normalization mean and std for each dim    |

## Data
Data can be downloaded from GitHub repository release page: TODO

| Path                      | Description                                        |
|---------------------------|----------------------------------------------------|
| `├── pid0/`               | Folder containing data for one person with `pid0`  |
| `├────── iid0.png`        | Image `iid0` that belongs to person `pid0`         |
| `├────── iid0.npz`        | StyleGAN2 latent vector of image `iid0`            |
| `├────── iid1.png`        |                                                    |
| `├────── iid1.npz`        |                                                    |
| `├── pid1/`               |                                                    |
| `├────── ...`             |                                                    |
| `├── ...`                 |                                                    |


## License
TODO