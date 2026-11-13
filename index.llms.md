# fast.ai Course

Working through both fast.ai courses: top-down practical deep learning first, then rebuilding the whole framework from scratch and ending at diffusion models.

Author

Benedict Thekkel

These are worked notebooks from the two fast.ai courses, which are deliberately taught in opposite directions.

Part 1 is top-down. You train and deploy a working model in the first lesson, then peel back layers to find out why it worked. Part 2 inverts that: it starts from matrix multiplication and rebuilds the training framework piece by piece, until the thing you have built is capable of training a diffusion model.

That second half is the reason this repo exports a library. Each foundations notebook adds to `fastAIcourse`, so by the end the package contains a small working deep learning framework assembled in course order.

------------------------------------------------------------------------

## Install

``` sh
pip install fastAIcourse
```

The library is the accumulated output of the Part 2 notebooks: `learner`, `activations`, `init`, `sgd`, `accel`, `resnet`, `augment`, `convolutions`, `datasets`, `training`, `ddpm`, `diffusion`, and `fid`. Each module comes from the notebook of the same name.

------------------------------------------------------------------------

## Part 1: Practical Deep Learning

Top-down. A working model first, explanations after.

| Page | Covers |
|----|----|
| [Getting Started](intro.llms.md) | Training the first model |
| [Deployment](deployment.llms.md) | Putting it somewhere people can use it, which comes second here rather than last |
| [Neural Net Foundations](neuralnet.llms.md) | Stochastic gradient descent, and why linear layers plus non-linear activations are flexible enough to fit anything |
| [How a Neural Net Really Works](how-does-a-neural-net-really-work.llms.md) | Fitting a function with gradient descent, worked slowly |
| [Natural Language Processing](naturallanguage.llms.md) | Text classification with pretrained language models |
| [Linear Model and Neural Net from Scratch](linear-model-and-neural-net-from-scratch.llms.md) | No pre-built architecture, optimizer, or data loading, to see what the framework was doing |
| [Why You Should Use a Framework](why-you-should-use-a-framework.llms.md) | The counterargument, immediately after |
| [How Random Forests Really Work](how-random-forests-really-work.llms.md) | The non-neural baseline that often wins on tabular data |
| [Kaggle Tutorial](kaggle_tut.llms.md) | Competing as a way of learning |
| [Multi-Target](multi-target.llms.md) | Predicting several things at once |
| [Collaborative Filtering](collaborative-filtering-deep-dive.llms.md) | Recommendation through learned embeddings |
| [Tabular Modeling](tabular.llms.md) | The long one, at 288 cells |
| [A Hacker’s Guide to Language Models](lm-hackers.llms.md) | What LLMs are and how to actually use them |

------------------------------------------------------------------------

## Stable Diffusion, Top Down

Part 2 opens by using diffusion models before explaining them.

| Page | Covers |
|----|----|
| [Overview](stable_diffusion_overview.llms.md) | What the pieces are |
| [Stable Diffusion with Diffusers](stable_diffusion.llms.md) | Driving it through Hugging Face, including attention slicing to trade speed for memory |
| [Deep Dive](stable_diffusion_deep_dive.llms.md) | Taking the pipeline apart |

------------------------------------------------------------------------

## Part 2: Foundations

Rebuilding the framework from the bottom. Most of these notebooks export into `fastAIcourse`.

| Page | Covers |
|----|----|
| [Matrix and Tensor](matrix.llms.md) | Matrix multiplication, from Python loops to broadcasting, at 234 cells |
| [Backpropagation](backpropagation.llms.md) | The chain rule, written out |
| [Mini-Batch Training](minibatch_training.llms.md) | The training loop taking shape |
| [Hugging Face Datasets](datasets.llms.md) | Getting data in |
| [Foundations](foundations.llms.md) | Callbacks and the machinery the learner needs |
| [Convolutions](convolutions.llms.md) | Building the operation, not calling it |
| [Learner](learner.llms.md) | The central abstraction that ties the loop together |
| [Activation Stats](activations.llms.md) | Instrumenting a model to see what its layers are doing |
| [Initialization](initializing.llms.md) | Why starting weights decide whether training works at all |
| [Accelerated SGD](accel_sgd.llms.md) | Momentum, RMSProp, Adam |
| [ResNets](resnet.llms.md) | Residual connections |
| [Augmentation](augment.llms.md) | Getting more out of the data you have |

------------------------------------------------------------------------

## Diffusion Models

Where Part 2 lands, once the framework exists to train them.

| Page | Covers |
|----|----|
| [DDPM with miniai](ddpm.llms.md) | Denoising diffusion probabilistic models, the first implementation |
| [DDPM v2](ddpm_v2.llms.md) | The second pass |
| [DDPM v3](ddpm_v3.llms.md) | The third |
| [DDIM](ddim.llms.md) | Implicit models, which sample in far fewer steps |
| [Karras Pre-conditioning](karras.llms.md) | The reformulation that made sampling schedules tractable |
| [Cosine Schedule](cosine.llms.md) | Choosing how noise is added over time |
| [Noise Prediction](noise-pred.llms.md) | Predicting how noisy a FashionMNIST image is |
| [Diffusion U-Net](diffusion_unet.llms.md) | The architecture doing the denoising |
| [Attention](attention.llms.md) | Adding attention to it |
| [Conditioned Diffusion](diffusion-attn-cond.llms.md) | Guiding generation with a condition |
| [FID](fid.llms.md) | Scoring generated images, since you cannot eyeball a distribution |

------------------------------------------------------------------------

## Experiments and Applications

| Page | Covers |
|----|----|
| [Style Transfer](styletransfer.llms.md) | Optimising an image against a style |
| [Neural Cellular Automata](nca.llms.md) | Growing an image from local update rules |
| [CIFAR-10 with Weights and Biases](cifar10_and_wandb.llms.md) | Classification with experiment tracking attached |
| [Tiny ImageNet](imgnet_tiny.llms.md) | The baseline run |
| [Tiny ImageNet, Widish](imgnet_tiny-widish.llms.md) | Wider, as a comparison |
| [Tiny ImageNet, Wide](imgnet_tiny-wide.llms.md) | Wider again |
| [Tiny ImageNet 200](imgnet_tiny-200.llms.md) | The baseline trained for 200 epochs |
| [Super-Resolution](superres.llms.md) | Upscaling |

------------------------------------------------------------------------

## Not Covered Yet

- **These are course notebooks, not a reference.** They follow the lessons, so coverage is whatever the course covered and the ordering is pedagogical rather than logical. For a reference treatment see [DL Methods](https://bthek1.github.io/DL_methods/).
- **No transformer or LLM training.** Part 2 ends at diffusion; attention appears only as a component of the U-Net.
- **Several pages have placeholder titles.** [Style Transfer](styletransfer.llms.md) and [NCA](nca.llms.md) are titled “Setup” and “Setup 2” internally, and the four Tiny ImageNet notebooks share one title, which is why the sidebar reads poorly.
- **The three DDPM versions are not distinguished** by their titles, so which to read is not obvious.
- **`180_autoencoder.ipynb` sits in the repo root rather than `nbs/`**, so it is neither published nor part of the library.
- The library has no documentation of its own beyond the notebooks that built it, so the exported API is only discoverable by reading them in order.

------------------------------------------------------------------------

Back to top
