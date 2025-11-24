# TODO

[x] 11/06 Dataloader 작성
[x] 11/08 `src/training/train_utils.py`의 데이터 로더가 `src/data/image_datasets.py`의 출력 형식과 맞지 않는 문제 수정
[ ]  Image concat 방식 변경
[x] 11/13 OCTA Dataset에 대해 학습
[ ] BerDiff에 prev. output의 thickness/SAUNA 대입
[ ] 11/24 BerDiff의 Bernoulli Sampling -> Poisson Sampling

## packages

```bash
uv pip install autorootcwd blobfile torch torchvision numpy \
    pillow mpi4py mpich \
    natsort matplotlib tqdm \
    pandas scikit-image
```
