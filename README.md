# Plantas50 image database

## Introduction
The Plantas image database is composed by 9,398 images in the Plantas50Basic subset, 1,277 in the Plantas50Extra subset, and 22,661 images in the Plantas50Internet subset for 50 different species and cultivars. The Plantas50Basic and Plantas50Extra set have high-quality images taken from digital cameras and smartphones. They have a resolution of 2048x1536, and were taken at gardens and parks in Brazil during the months of December/2015 and March/2016.

## Download
You can use the python downloader that will automatically download the whole database to your computer. The total size is about 16GB and you should have at least 32GB free for the download and merge operations.

    To save the dataset in the same directory of the script (`files_url_hashes.txt` also must be in the same directory), run:
    `python download_plantas50.py`
    
    To save in another directory, run:
    `python download_plantas50.py /path/to/files_url_hashes.txt /path/to/dir`

You can also access: <https://1drv.ms/f/s!AjZCiYkckpt_g-YzIOQ4FWiegQXVtQ> and download from there.
Then you can run `cat Plantas50.tar.part* > Plantas50.tar` or in Windows `copy /b Plantas50.tar.part* Plantas50.tar` and you'll be ready to go!

BY DOWNLOADING YOU ACCEPT TO USE THE INTERNET SUBSET FOR RESEARCH OR NON-COMMERCIAL USE ONLY. SEE LEGAL SECTION.

## Supplemental Material

Supplemental Material for the paper 'Visual Recognition of Plant Species in the Wild' can be found in the paper-data folder.

## Details

| Label                               | Basic | Extra | Internet | All  |
|-------------------------------------|------:|------:|---------:|-----:|
| Agave americana 'Marginata'         | 201  | 0        | 240      | 441  |
| Agave angustifolia                  | 236  | 4        | 370      | 610  |
| Agave attenuata                     | 200  | 64       | 1036     | 1300 |
| Agave ovatifolia                    | 203  | 3        | 318      | 524  |
| Allamanda blanchetii                | 201  | 0        | 380      | 581  |
| Allamanda cathartica                | 200  | 4        | 1002     | 1206 |
| Alpinia purpurata                   | 201  | 0        | 1137     | 1338 |
| Anthurium andraeanum                | 201  | 120      | 760      | 1081 |
| Beaucarnea recurvata                | 191  | 11       | 763      | 965  |
| Begonia × hybrida                   | 100  | 0        | 226      | 326  |
| Bismarckia nobilis                  | 197  | 4        | 556      | 757  |
| Bougainvillea glabra                | 196  | 4        | 726      | 926  |
| Buxus microphylla                   | 219  | 24       | 195      | 438  |
| Callistemon spp                     | 202  | 4        | 415      | 621  |
| Clerodendrum × speciosum            | 201  | 63       | 174      | 438  |
| Codiaeum variegatum 'Aureo-maculatum' | 221  | 0        | 87       | 308  |
| Cordyline fruticosa                 | 201  | 1        | 551      | 753  |
| Cupressus sempervirens              | 200  | 85       | 315      | 600  |
| Cycas revoluta                      | 221  | 91       | 833      | 1145 |
| Cycas thouarsii                     | 203  | 4        | 225      | 432  |
| Davallia fejeensis                  | 200  | 77       | 136      | 413  |
| Dianella ensifolia                  | 199  | 3        | 211      | 413  |
| Dieffenbachia amoena                | 88   | 3        | 218      | 309  |
| Dracaena marginata                  | 115  | 88       | 248      | 451  |
| Duranta erecta 'Gold Mound'         | 200  | 0        | 89       | 289  |
| Dypsis lutescens                    | 203  | 89       | 410      | 702  |
| Echeveria glauca                    | 200  | 114      | 290      | 604  |
| Eugenia sprengelii                  | 209  | 23       | 26       | 258  |
| Hibiscus rosa-sinensis              | 211  | 3        | 1208     | 1422 |
| Impatiens hawkeri                   | 100  | 0        | 549      | 649  |
| Ixora coccinea                      | 200  | 32       | 617      | 849  |
| Ixora coccinea 'Compacta'           | 201  | 117      | 215      | 533  |
| Justicia brandegeana                | 214  | 0        | 600      | 814  |
| Leea guineensis                     | 100  | 5        | 59       | 164  |
| Loropetalum chinense                | 204  | 2        | 729      | 935  |
| Monstera deliciosa                  | 220  | 0        | 806      | 1026 |
| Nematanthus wettsteinii             | 191  | 48       | 97       | 336  |
| Nerium oleander                     | 195  | 12       | 1236     | 1443 |
| Ophiopogon jaburan                  | 210  | 6        | 123      | 339  |
| Philodendron imbe                   | 225  | 0        | 20       | 245  |
| Philodendron martianum              | 97   | 3        | 120      | 220  |
| Phoenix roebelenii                  | 213  | 2        | 330      | 545  |
| Podocarpus macrophyllus             | 98   | 2        | 374      | 474  |
| Rhapis excelsa                      | 201  | 75       | 687      | 963  |
| Rhododendron simsii                 | 218  | 0        | 485      | 703  |
| Russelia equisetiformis             | 205  | 9        | 641      | 855  |
| Strelitzia reginae                  | 200  | 67       | 1075     | 1342 |
| Syngonium angustatum                | 198  | 2        | 38       | 238  |
| Zamioculcas zamiifolia              | 97   | 3        | 490      | 590  |
| Zinnia peruviana                    | 191  | 7        | 225      | 423  |


## Legal

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Plantas50Basic and Plantas50Extra</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/reneoctavio" property="cc:attributionName" rel="cc:attributionURL">Rene Octavio Queiroz Dias</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Some images of Plantas50Internet subset may have copyright. Training and using recognition model for research or non-commercial use may constitute fair use of data.

All code is under MIT license, unless stated otherwise in the header of the code.

## Citation

If the Plantas database was useful in your publications, please cite:

    @inproceedings{plantas-db-2016,
    author={Dias, Ren{\'e} Octavio Queiroz and Borges, D{\'i}bio Leandro},
    booktitle={2016 IEEE International Symposium on Multimedia (ISM)},
    title={Recognizing Plant Species in the Wild: Deep Learning Results and a New Database},
    year={2016},
    pages={197-202},
    doi={10.1109/ISM.2016.0047},
    isbn={978-1-5090-4571-6/16},
    url={https://doi.org/10.1109/ISM.2016.0047},
    month={Dec},}
