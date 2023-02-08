<h1 align="center">
  ToolFlowNet: Robotic Manipulation with Tools via Predicting Tool Flow from Point Clouds</h1>

<div align="center">
  <a href="http://www.cs.cmu.edu/~dseita/">Daniel Seita</a> &nbsp;•&nbsp;
  <a href="https://yufeiwang63.github.io/">Yufei Wang<sup>†</sup></a> &nbsp;•&nbsp;
  <a href="https://sarthakjshetty.github.io/">Sarthak J. Shetty<sup>†</sup></a> &nbsp;•&nbsp;
  Edward Y. Li<sup>†</sup> &nbsp;•&nbsp;
  <a href="https://zackory.com/">Zackory Erickson</a> &nbsp;•&nbsp;
  <a href="https://davheld.github.io/">David Held</a>
</div>

<h4 align="center">
  <a href="https://sites.google.com/view/point-cloud-policy/home"><b>Website</b></a> &nbsp;•&nbsp;
  <a href="https://arxiv.org/abs/2211.09006"><b>Paper</b></a>
</h4>

<hr>

:warning: **This is the branch for training ToolFlowNet on the real world robot data** :warning:

This has the **ToolFlowNet** code for
<a href="https://arxiv.org/abs/2211.09006"><b>our CoRL 2022 paper</b></a>.
See <a href="https://github.com/DanielTakeshi/softgym_tfn">our other repository</a>
for the SoftGym environment code. Note that you should install that code *first*,
and then this ToolFlowNet code *second*. Furthermore, this branch specifically is for training and evaluating ToolFlowNet on the real world data.

The code in this repository has a bunch of SoftGym-specific components. We are
working on a separate, simplified version of ToolFlowNet which is more agnostic
to the choice of environment or task.

<hr>

Contents:

- [Installation](#installation)
- [How to Use](#how-to-use)
- [CoRL 2022 Experiments](#corl-2022-experiments)
- [Citation](#citation)

<hr>

Running the instructions in this branch assumes that you have correctly installed [`softgym_tfn`](https://github.com/DanielTakeshi/softgym_tfn) and `softagent_tfn`[https://github.com/DanielTakeshi/softagent_tfn/].

## Installation

Assuming you have already installed the other repository, then for this one, we
first need to make a symlink to `softgym_tfn`. Make sure that `softgym_tfn/`
does not exist within this folder. Then run this:

```
ln -s ../softgym_tfn/ softgym
```

This creates a symlink so the `softgym` subdirectory points to our `softgym_tfn`
repository. By typing in `ls -lh` you should see: `softgym -> ../softgym_tfn/`.

Then run this in any new Terminal window/tab that you want to run code in:

```
. ./prepare_1.0.sh
```

This will go into `softgym` and set up the conda environment. It should also set
`PYFLEXROOT` which points to `PyFlex` in the SoftAgent repository.

## How to Use

The main way that we launch experiments is with:

```
python launch_exp.py
```

Running this command will launch the training loop. Make sure you check the 

See `launch_exp.py` for details on what to modify. The three main areas to
adjust for the purpose of learning from demonstrations are:

- Adjusting the behavioral cloning data directory.
- Make sure that the right method has been selected in the `launch_exp.py` file. 

You can find these areas by searching in `launch_exp.py` for this pattern:

```
# ----------------------------- ADJUST -------------------------------- #

# --------------------------------------------------------------------- #
```

Check the content in between the above two lines.

See the next section for how we set these for the CoRL 2022 submission.

**Important note**: we highly recommend using `wandb` to track experiments.
Please adjust these two lines in `launch_exp.py` appropriately:

```
vg.add('wandb_project', [''])  # Fill this in!
vg.add('wandb_entity', [''])  # Fill this in!
```

If you need a refresher on `wandb`, refer to [the official documentation][2]. If
you leave these blank, the script might not run successfully.

## CoRL 2022 Experiments

Before this, make sure you have downloaded demonstration data [following our
other repository's instructions][1]. This includes both the cache and the
demonstrations themselves. You may put the data in a different location if
desired.

**First**: with the demonstration data, adjust the `DATA_HEAD` variable. For
example, setting this:

```
DATA_HEAD = '/data/sarthak/softagent_tfn_physical/data_demo/'
```

near the top of `launch_exp.py` means that, I should expect to see the demonstrations located at:

```
/data/sarthak/softagent_tfn_physical/data_demo/v07_rotation_translation_variably_composed
```

Once you are confident the settings are correct, run the script! (Did you
remember to set up `wandb`?)

## Citation

If you find this repository useful, please cite our paper:

```
@inproceedings{Seita2022toolflownet,
    title={{ToolFlowNet: Robotic Manipulation with Tools via Predicting Tool Flow from Point Clouds}},
    author={Seita, Daniel and Wang, Yufei and Shetty, Sarthak, and Li, Edward and Erickson, Zackory and Held, David},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2022}
}
```

[1]:https://github.com/SarthakJShetty/tfn-robot
[2]:https://docs.wandb.ai/