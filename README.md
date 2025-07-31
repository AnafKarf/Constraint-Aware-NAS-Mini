
# Constraint-Aware NAS-Mini
The repository was cloned from [HyeonjeongHa/CRoZe](https://github.com/HyeonjeongHa/CRoZe)

## Installation 
- `python == 3.8.12`
```sh
pip install -r requirements.txt
```
### Preparation
Locate below two benchmarks in `./data` directory
- [Robust NAS-Bench-201](https://uni-siegen.sciebo.de/s/aFzpxCvTDWknpMA)
- [NAS-Bench-201](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view)
- [Common-corruption]()
- [ImageNet16-120](https://drive.google.com/drive/folders/1NE63Vdo2Nia0V7LK1CdybRLjBFY72w40)

## NAS-Bench-201
### Spearman's rank correlation
```sh
# sh scripts/nb201/corr.sh
CUDA_VISIBLE_DEVICES=0 python main.py --search_space nasbench201 --proxy_types 'baselines/croze' --api_loc 'PATH_TO_API' --rob_api_loc 'PATH_TO_ROBUST_API' --data 'PATH_TO_DATA' --dataset 'cifar10/cifar100/ImageNet16-120' --save_dir 'PATH_TO_SAVE' --start '0' --end '100/15625'
```
- Clean zero-shot NAS baselines (`gradnorm, grasp, fisher, synflow, plain`) can be downlowded [here](https://drive.google.com/drive/folders/1mSKVpH5vqTB1shrKnraKDJy_983dEyQJ).
- Our zero-shot NAS results can be found in `./results` directory.
- Calculate Spearman's rank correlation directly as below if you already have zero-cost proxy values.
```sh
python corr.py --search_space nasbench201 --proxy_types 'baselines/croze' --rob_api_loc 'PATH_TO_ROBUST_API' --dataset 'cifar10/cifar100/ImageNet16-120' --save_dir 'PATH_TO_SAVE' --start '0' --end '100/15625' --proxy_path 'PATH_TO_PROXY_RESULTS'
```

## DARTS
### Architecture Search 
```sh
# sh scripts/darts/search.sh
CUDA_VISIBLE_DEVICES=0 python main.py --search_space 'darts' --w_sampling --sampling_type'mutate' --sample_num 'SAMPLE_NUM' --sample_pool 'SAMPLE_POOL' --init_pool 'INIT_POOL' --proxy_types 'croze' --save_dir 'PATH_TO_SAVE' --data 'PATH_TO_DATA' --dataset 'cifar10/cifar100/ImageNet16-120' 
```
- If you want to conduct end-to-end experiments, use `--e2e --adv_loss ' /pgd'`. `''` do standard training and `pgd` do adversarial trainig.

### Architecture Evaluation 
- Standard training
```sh
CUDA_VISIBLE_DEVICES=0 python train.py --search_space darts --arch_type croze_cifar10 --adv_loss '' --save_dir 'PATH_TO_SAVE' --data 'PATH_TO_DATA' --dataset 'cifar10/cifar100/ImageNet16-120' 
```
- Adversarial training
```sh
CUDA_VISIBLE_DEVICES=0 python train.py --search_space darts --arch_type croze_cifar10 --adv_loss 'pgd' --save_dir 'PATH_TO_SAVE' --data 'PATH_TO_DATA' --dataset 'cifar10/cifar100/ImageNet16-120' 
```
- Evaluation
```sh
CUDA_VISIBLE_DEVICES=0 python train.py --search_space darts --arch_type croze_cifar10 --eval_only --eval_cc --eval_attack_type 'pgd/cw/deepfool/spsa/lgv/autoattack' --save_dir 'PATH_TO_SAVE' --data 'PATH_TO_DATA' --dataset 'cifar10/cifar100/ImageNet16-120' 
```
### Reproducing Results 
```sh
CUDA_VISIBLE_DEVICES=0 python train.py --search_space darts --arch_type croze_cifar10 --adv_loss 'pgd/ ' --save_dir 'PATH_TO_SAVE' --data 'PATH_TO_DATA' --dataset 'cifar10' 
```
## Citation
If you found the provided code useful, please cite our work.
```
@article{ha2023generalizable,
  title={Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations},
  author={Ha, Hyeonjeong and Kim, Minseon and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2306.05031},
  year={2023}
}
```


