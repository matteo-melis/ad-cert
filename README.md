# Adversarial Distillation for Certification (AD-CERT)

This repository contains the code used for our paper, [**Improving Certified Robustness via Adversarial Distillation**](https://arxiv.org/abs/2606.31653), which has been accepted into the proceedings of BMVC 2026. The codebase is a local copy of the unified certified training library [**CTBench**](https://github.com/eth-sri/CTBench). Hence, most of the training, model loading and certification infrastructure is inherited directly from the original CTBench implementation.

The main additions in this submission are support for:

- **AD-CERT**, enabled with `--use-adcert-training`, and
- **CC-DIST**, enabled with `--use-ccdist-training`

With the following being added also:

- **Cyclic learning-rate schedules**, enabled with `--lr-schedule cyclic`
- **Soft-label adversarial examples**, enabled with `--use-adcert-training, --use-softlabel-attack`
- **RSLAD-CERT**, enabled with `--use-adcert-training, --use-rslad`
- **NAT-KL**, enabled with `--use-adcert-training, --use-nat-kl`

## Main implementation file changes

We add our main changes in the following files:

- `args_factory.py`: adds command line args for AD-CERT and CC-DIST.
- `model_wrapper.py`: implements the AD-CERT and CC-DIST training objectives.
- `mix_train.py`: adds support for selecting AD-CERT/CC-DIST training and cyclic learning-rate scheduling.
- `attacks.py`: contains small additions needed for teacher-guided adversarial examples.

## Reproducibility and pretrained models

For reproducibility, we include example scripts for training and certification, as well as our pretrained AD-CERT models inside an `AD_CERT_RELEASE` folder on a [MEGA drive](https://mega.nz/file/XShSXbAS#DFsjICl8XHtx__xaEKkNCDNBbHLB-YjbjwH1T7rCNis). Please note this requires ~1GB of memory.

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

## Notes

This repository is intended to document the additions required for AD-CERT and the experiments in our paper. It is not a standalone reimplementation of CTBench. For general usage, environment setup, supported baselines, and certification details, please refer to the original CTBench documentation included in this repository.
