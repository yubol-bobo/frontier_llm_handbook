# Source snapshots

记录每份初读笔记对应的源码版本。不是跨项目兼容性 lockfile；实际训练仍应使用各项目自己的依赖锁定。

所有源码为 shallow clone，保留当前工作树；不含完整历史、未初始化的 submodules 或 Git LFS 大文件内容。模型权重与数据集不在本目录。

| 项目 | 本地源码 | 固定版本 | 文件数 | Submodules | LFS pointers |
|---|---|---|---:|---:|---:|
| Pi | [sources/pi](sources/pi/) | [6160683a4a80](https://github.com/earendil-works/pi/tree/6160683a4a8012f0d1cd30c145df18b4ca6f5176) | 1697 | 0 | 0 |
| DeepSeek Harness | [sources/deepseek-harness](sources/deepseek-harness/) | [c389f96bf3a9](https://github.com/deepseek-ai/deepseek-harness/tree/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8) | 9599 | 0 | 0 |
| Harbor Cookbook | [sources/harbor-cookbook](sources/harbor-cookbook/) | [e093c9a860b9](https://github.com/harbor-framework/harbor-cookbook/tree/e093c9a860b988d9d74901010ddddb9c7f124f92) | 124 | 0 | 0 |
| Harbor | [sources/harbor](sources/harbor/) | [9a2e3b135cc8](https://github.com/harbor-framework/harbor/tree/9a2e3b135cc8fb1e41131020f83370cb2f12ae93) | 3431 | 0 | 0 |
| Verifiers | [sources/verifiers](sources/verifiers/) | [27bbd216df0a](https://github.com/PrimeIntellect-ai/verifiers/tree/27bbd216df0af719a43705866b2cf6139bcc95de) | 354 | 0 | 0 |
| Prime RL | [sources/prime-rl](sources/prime-rl/) | [04a61d3b75c3](https://github.com/PrimeIntellect-ai/prime-rl/tree/04a61d3b75c3c99f263b2c133e822f998909adf7) | 556 | 5 | 0 |
| APEX Agents SkyRL Recipe | [sources/apex-agents-skyrl-recipe](sources/apex-agents-skyrl-recipe/) | [8e7702f03b74](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/tree/8e7702f03b7464a36ab800a624fd911de0968a87) | 30 | 0 | 0 |
| SmolLM | [sources/smollm](sources/smollm/) | [a041759883ec](https://github.com/huggingface/smollm/tree/a041759883ec7152d18fb985ea49be641a0bceef) | 711 | 0 | 0 |
| OLMo-core | [sources/olmo-core](sources/olmo-core/) | [92870a33c3fe](https://github.com/allenai/OLMo-core/tree/92870a33c3fee060d57c3faec52b0b369ece85a6) | 531 | 0 | 0 |
| Open Instruct | [sources/open-instruct](sources/open-instruct/) | [ebd0c8e0c1d7](https://github.com/allenai/open-instruct/tree/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5) | 751 | 0 | 4 |
| Marin | [sources/marin](sources/marin/) | [5e2436d0f614](https://github.com/marin-community/marin/tree/5e2436d0f61462983003bd8b6eaef8235ecab78c) | 3831 | 0 | 0 |
| TorchTitan | [sources/torchtitan](sources/torchtitan/) | [4e20e76b235e](https://github.com/pytorch/torchtitan/tree/4e20e76b235ec415bd81912ce09eed46fa39f717) | 739 | 0 | 0 |
| Megatron-LM / Megatron Core | [sources/megatron-lm](sources/megatron-lm/) | [c9b53d0a87cb](https://github.com/NVIDIA/Megatron-LM/tree/c9b53d0a87cb926f47115259593ecfeb351ca29f) | 3216 | 0 | 0 |
| verl | [sources/verl](sources/verl/) | [7cb65014d3a6](https://github.com/verl-project/verl/tree/7cb65014d3a6c84f59458367df768999e4f36c67) | 1224 | 1 | 0 |
| slime | [sources/slime](sources/slime/) | [4c193f1f3750](https://github.com/THUDM/slime/tree/4c193f1f37509cca70f0e88807a9305b70f63f4e) | 608 | 0 | 0 |
| Miles | [sources/miles](sources/miles/) | [3de96596f16b](https://github.com/radixark/miles/tree/3de96596f16b9e6d23ba550c4c47de3479c9f14c) | 2052 | 0 | 0 |
| SGLang | [sources/sglang](sources/sglang/) | [30e7a3072d3f](https://github.com/sgl-project/sglang/tree/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4) | 8968 | 0 | 0 |
| DeepGEMM | [sources/deepgemm](sources/deepgemm/) | [559d79fb6994](https://github.com/deepseek-ai/DeepGEMM/tree/559d79fb6994a58b8a15b4b93bf13ccc16edf247) | 138 | 2 | 0 |
| DeepEP | [sources/deepep](sources/deepep/) | [01dc3aaac820](https://github.com/deepseek-ai/DeepEP/tree/01dc3aaac82068020353dce2c302e38153c0bfaa) | 108 | 1 | 0 |

## 使用说明

- 本地源码互相独立，学习仓库通过 `.gitignore` 排除 `sources/`，避免嵌套仓库误提交。
- 学习笔记引用固定 SHA。之后手工更新源码时，应新建学习记录并重新记录快照，不覆盖过去实验依据。
- `python tools/clone_repos.py` 只克隆缺失仓库并检查已有 origin，不 pull/reset 已有源码。
- `python tools/snapshot_sources.py` 重新记录当前版本；只有在需要更新来源清单时才执行。
- Submodule 仅记录 gitlink；需要编译某个项目时再按其说明获取对应依赖。
