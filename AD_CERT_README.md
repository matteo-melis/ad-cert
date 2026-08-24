# Adversarial Distillation for Certification (AD-CERT)

This repository contains the code used for BMVC 2026 submission number 1821, '**Improving Certified Robustness via Adversarial Distillation**'. The codebase is a local copy of the unified certified training library **CTBench**. Hence, most of the training, model loading and certification infrastructure is inherited directly from the original CTBench implementation.

The main additions in this submission are support for:

- **AD-CERT**, enabled with `--use-adcert-training`, and
- **CC-DIST**, enabled with `--use-ccdist-training`

With the following being added also:

- **Cyclic learning-rate schedules**, enabled with `--lr-schedule cyclic`
- **Soft-label adversarial examples**, enabled with `--use-adcert-training, --use-softlabel-attack`
- **RSLAD-CERT**, enabled with `--use-adcert-training, --use-rslad`
- **NAT-KL**, enabled with `--use-adcert-training, --use-nat-kl`

## Main implementation files

We add our main changes in the following files:

- `args_factory.py`: adds command line args for AD-CERT and CC-DIST.
- `model_wrapper.py`: implements the AD-CERT and CC-DIST training objectives.
- `mix_train.py`: adds support for selecting AD-CERT/CC-DIST training and cyclic learning-rate scheduling.
- `attacks.py`: contains small additions needed for teacher-guided adversarial examples.

## Training and Certification of AD-CERT

For reproducibility, we include example scripts for training and certification of AD-CERT inside the `AD_CERT_RELEASE` folder.

## AD-CERT Pretrained Models

We have uploaded our pretrained models to MEGA. Please note this requires xGB of memory.

## Notes

This repository is intended to document the additions required for AD-CERT and the experiments in our paper. It is not a standalone reimplementation of CTBench. For general usage, environment setup, supported baselines, and certification details, please refer to the original CTBench documentation included in this repository.

## Citing our work

If you use our work in anyway, please kindly cite our paper:

```
@inproceedings{melis2026adcert,
      title={Improving Certified Robustness via Adversarial Distillation},
      author={Matteo Melis and Jesus Martinez Del Rincon and Vishal Sharma},
      booktitle={British Machine Vision Conference},
      year={2026}
}
```
