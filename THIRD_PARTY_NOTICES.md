# Third-Party Notices

Unless otherwise noted, original project code is licensed under the repository's [Apache License 2.0](./LICENSE). The adapted components below retain their upstream licenses and have been modified for this project.

| Component | Local scope | License |
|---|---|---|
| [PixelDiT](https://github.com/NVlabs/PixelDiT) | `imgnet/networks/backbone/pixeldit/{modules.py,model.py}` | [NVIDIA NSCLv1](./LICENSES/PixelDiT-NSCLv1.txt) |
| [PixNerd](https://github.com/MCG-NJU/PixNerd) | portions of the PixelDiT modules above, attributed by its [upstream NOTICE](./LICENSES/PixelDiT-NOTICE.txt) | [MIT](./LICENSES/MIT-NOTICES.txt) |
| [FD-Loss](https://github.com/Jiawei-Yang/FD-Loss) | adapted portions of `imgnet/networks/backbone/jit/{modules.py,denoiser.py,model.py}` and `imgnet/metrics/{eval_all_fds.py,frechet_distance/**,utils/**}` | [MIT](./LICENSES/MIT-NOTICES.txt) |
| [JiT](https://github.com/LTH14/JiT) | JiT-derived portions of `imgnet/networks/backbone/jit/{modules.py,model.py}` | [MIT](./LICENSES/MIT-NOTICES.txt) |
| [EVA-02](https://github.com/baaivision/EVA) / [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch) | `imgnet/networks/backbone/dit/pos_embed.py` | [MIT](./LICENSES/MIT-NOTICES.txt) |
| [UCGM](https://github.com/LINs-lab/UCGM) | `imgnet/networks/backbone/dit/model.py`, adapted from its TiT implementation developed by Peng Sun | [Apache-2.0](./LICENSE) |
| [torch-fidelity](https://github.com/toshas/torch-fidelity) / [pytorch-fid](https://github.com/mseitzer/pytorch-fid) | Inception implementation in `imgnet/metrics/utils/perception_util.py` (torch-fidelity: Copyright 2020 Anton Obukhov) | [Apache-2.0](./LICENSE) |
| [torchvision](https://github.com/pytorch/vision) | upstream Inception blocks in `imgnet/metrics/utils/perception_util.py` | [BSD-3-Clause](./LICENSES/Torchvision-BSD-3-Clause.txt) |
| [Meta DINOv2](https://github.com/facebookresearch/dinov2) | `imgnet/networks/backbone/dit/swiglu_ffn.py` | [Apache-2.0](./LICENSE) |
| [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | `center_crop_arr` in `imgnet/networks/representors/utils.py`; reference implementation for `compute_fid` in `imgnet/metrics/frechet_distance/metrics.py` | [MIT](./LICENSES/MIT-NOTICES.txt) |

The PixelDiT files above and their modifications are governed by NSCLv1, not Apache-2.0. Its [upstream NOTICE](./LICENSES/PixelDiT-NOTICE.txt), including Copyright (c) NVIDIA Corporation, is retained; the local scope table identifies the vendored files. NSCLv1 permits their use only for non-commercial research or evaluation. Copyright and permission notices for MIT-licensed components are consolidated in [MIT-NOTICES.txt](./LICENSES/MIT-NOTICES.txt).

External model weights, datasets, and separately installed dependencies remain subject to their respective upstream terms.
