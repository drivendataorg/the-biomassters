
<img alt="An image of a forest in Finland" src="https://drivendata-public-assets.s3.amazonaws.com/biomass-finnish-forests.jpg" style="
    object-fit: cover;
    object-position: center;
    height: 250px;
    width: 100%;
">

<br />

# The BioMassters

## Goal of the competition
In this challenge, the competitors' goal was to build a model to predict the yearly [Aboveground Biomass (AGBM)](https://www.un-redd.org/glossary/aboveground-biomass) for 2,560 x 2,560 meter patches of Finnish forests using satellite imagery from [Sentinel-1 (S1)](https://sentinel.esa.int/web/sentinel/missions/sentinel-1) and [Sentinel-2 (S2)](https://sentinel.esa.int/web/sentinel/missions/sentinel-2). AGBM is a widespread metric for the study of [carbon release and sequestration by forests](https://unece.org/forests/carbon-sinks-and-sequestration) and is used by forest owners, policymakers, and conservationists to make decisions about forest management. 

## What's in this repository

This repository contains code from winning competitors in the [BioMassters](https://www.drivendata.org/competitions/99/biomass-estimation/page/534/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning submissions

Place | User | Private Score | Summary of Model
--- | --- | ---   | ---
1  | [kbrodt](https://www.drivendata.org/users/kbrodt/) | 27.63 | Combined S1 and S2 images into a 15-band 12-month composite, used a UNET model with test time augmentation.
2   | Team Just4Fun: [qqggg](https://www.drivendata.org/users/qqggg/), [HongweiFan](https://www.drivendata.org/users/HongweiFan/) | 27.68 | Used a SWIN UNETR model adopted from the [Medical Open Network for AI](https://monai.io/) on satellite features represented in 3D.
3   | [yurithefury](https://www.drivendata.org/users/yurithefury/) | 28.04 | Aggregated S1 and S2 data into 6 median composites per year, ensembled together 15 models using UNET++ architecture.
MATLAB Bonus Prize | Team D_R_K_A: [kaveh9877](https://www.drivendata.org/users/kaveh9877/), [AZK90](https://www.drivendata.org/users/AZK90/) | 31.08 | Combined S1 and S2 images into a 15-band 12-month composite, used a 1-D CNN to perform by-pixel regression. The resulting labels were added as a 16th band to the composites, which were then passed through a 3-D U-Net model.  

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark blog post: [The BioMassters](https://drivendata.co/blog/biomass-benchmark)**
