# LME Cluster Guide

The LME cluster is a GPU cluster operated by Computer Science Chair 5 (CS5 / I5)
at FAU Erlangen-Nürnberg. It is intended for heavy GPU workloads — not for
CPU-only jobs — and is managed with [Slurm](https://slurm.schedmd.com/).

A useful companion repo with HPC tips:
<https://gitos.rrze.fau.de/ym60imaq/i5_cluster_onboarding>.

If you want to give a student access, consider the RRZE HPC cluster first
(also Slurm-based) — that frees up LME resources.

## Connecting

```bash
ssh <your_ldap_id>@cluster.i5.informatik.uni-erlangen.de
```

This puts you on the submit node `lme242`. You don't run jobs there directly —
you submit them to Slurm and the scheduler runs them on a compute node when the
requested resources are free.

## Mailing list and support

Subscribe to the `cs5-cluster` mailing list before you start using the cluster.

For problems, write to <cs5-admin-cluster@lists.fau.de> — but check this
document and ask your advisor first.

## User guidelines

The cluster is shared. Following these rules keeps it usable for everyone.

1. **Use `/scratch` for hot data.** Move data from `/cluster/<user>/data` to
   `/scratch/<user>` on the compute node before training. This reduces shared-FS
   bandwidth and is faster (local SSD). `/scratch` is per-node and ephemeral —
   copy data in at job start (the example script below shows how).
2. **Implement checkpointing for long jobs.** A 24-hour wall-time limit is
   enforced; submit a follow-up job once you hit it. Extensions are possible in
   rare cases — ask the admins.
3. **Clean up.** Remove unused models and datasets so `/cluster` doesn't fill
   up.
4. **Don't hold idle GPUs.** Make sure jobs that request GPUs actually use
   them. Monitor with `nvtop` (see [Analyse GPU usage on a specific node](#analyse-gpu-usage-on-a-specific-node)).

## Basic concepts

You don't compute on the submit node — you submit jobs from it, and Slurm
schedules them on a compute node. As soon as a node has the requested resources
free (and you're under your concurrent-job limit), the job runs.

Useful Slurm references:

- <https://hpc.fau.de/systems-services/systems-documentation-instructions/batch-processing/>
- <https://www.fau.tv/clip/id/41306>

## Hardware

Compute nodes: `lme49`, `lme50`, `lme51`, `lme52`, `lme53`, `lme170`, `lme171`,
`lme221`, `lme222`, `lme223`.

| Node    | GPUs                                                          |
|---------|---------------------------------------------------------------|
| lme49   | 1× Quadro RTX A6000 (48.6 GiB), 2× GeForce GTX 1080 (8.1 GiB) |
| lme50   | 1× Titan XP (12.1 GiB), 3× GeForce GTX 1080 Ti (11.1 GiB)     |
| lme51   | 4× GeForce GTX 1080 Ti (11.1 GiB)                             |
| lme52   | 4× GeForce GTX 1080 Ti (11.1 GiB)                             |
| lme53   | 4× NVIDIA Tesla V100 SXM2 (16.1 GiB)                          |
| lme170  | 2× Quadro RTX 8000 (48.6 GiB)                                 |
| lme171  | 2× Quadro RTX 8000 (48.6 GiB)                                 |
| lme221  | 4× Quadro RTX 6000 (24.2 GiB)                                 |
| lme222  | 4× Quadro RTX 5000 (16.1 GiB)                                 |
| lme223  | 4× Quadro RTX 5000 (16.1 GiB)                                 |

Query GPUs per node:

```bash
sinfo -h -o "%n %G"
```

Slurm-internal GPU type identifiers (output of the above):

```
lme49  gpu:gtx1080:4
lme50  gpu:titanxp:1,gpu:gtx1080ti:3
lme51  gpu:gtx1080ti:4
lme52  gpu:gtx1080ti:4
lme53  gpu:teslav100sxm216gb:4
lme170 gpu:q8000:2
lme171 gpu:q8000:2
lme221 gpu:q6000:4
lme222 gpu:q5000:4
lme223 gpu:q5000:4
```

Use those identifiers to request specific GPUs:

- `--gres=gpu:1` — any single GPU
- `--gres=gpu:q5000:1` — one Quadro RTX 5000

## Datasets

Put large datasets in `/cluster/shared_data` so other users can reuse them.

## Job submission

Submit from `cluster.i5.informatik.uni-erlangen.de` (`lme242`). The standard
flow is to write a small shell script with `#SBATCH` directives at the top, then
call `sbatch <script.sh>`.

```bash
#!/bin/bash
#SBATCH --job-name=MY_EXAMPLE_JOB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/%x-%j-on-%N.out
#SBATCH -e /home/%u/%x-%j-on-%N.err
#SBATCH --mail-type=ALL
# Time limit format: hours:minutes:seconds — max is 24h
#SBATCH --time=24:00:00
#SBATCH --exclude=lme53
# To pin a GPU type: #SBATCH --gres=gpu:q5000:1
# Run `sinfo -h -o "%n %G"` for GPU types

# Tell pipenv to install virtualenvs on the cluster filesystem
export WORKON_HOME=/cluster/$(whoami)/.python_cache

echo "Your job is running on $(hostname)"

# Where possible, stage data from /cluster (network FS) to /scratch (local SSD).
cp -r /cluster/<your_name>/data /scratch/$SLURM_JOB_ID/data

# Or untar an archive on the fly (create with: tar -cf data.tar data):
# tar -xf /cluster/<your_name>/data.tar -C /scratch/$SLURM_JOB_ID/

# Small Python packages can go in your home dir; use pipenv (below) for big ones.
pip3 install --user -r cluster_requirements.txt
python3 train.py
```

All `#SBATCH` directives must be at the top of the file. The example reserves
1 task, 2 CPUs, 12 GB RAM, and 1 GPU. See `man sbatch` for more options.

You can get an interactive shell with `srun --pty --nodelist=lme49 bash -i` —
but please avoid this for routine work.

### Chain multiple jobs

Submit jobs with `--dependency afterok:<jobid>` so each starts only after the
previous one succeeds:

```bash
#!/bin/bash
TASKS="pre-processing.sl mpi.sl post-processing.sl"
DEPENDENCY=""
for TASK in $TASKS; do
    JOB_CMD="sbatch"
    if [ -n "$DEPENDENCY" ]; then
        JOB_CMD="$JOB_CMD --dependency afterok:$DEPENDENCY"
    fi
    JOB_CMD="$JOB_CMD $TASK"
    echo -n "Running command: $JOB_CMD  "
    OUT=$($JOB_CMD)
    echo "Result: $OUT"
    DEPENDENCY=$(echo $OUT | awk '{print $4}')
done
```

More: <https://github.com/HPCNow/hpcnow-labs/blob/master/user-training/05-setting-up-complex-workflows.md>

## Python environments

### Miniconda

System Miniconda lives at `/opt/miniconda` on compute nodes (not on `lme242`).
Add to your job script:

```bash
export PATH=/opt/miniconda/bin:$PATH
```

Use `pip --user` to install. Re-direct the conda package cache off `/home`:

```bash
export CONDA_PKGS_DIRS="/cluster/$(whoami)/miniconda/pkgs"
```

If the system Miniconda is unavailable, install your own under `/cluster`:

```bash
cd /cluster/$(whoami)
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -u -p /cluster/$(whoami)/miniconda
```

Wire conda into your shell:

```bash
eval "$(/cluster/$(whoami)/miniconda/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

If you don't modify `~/.bashrc`, add this to your job script instead:

```bash
export PATH="/cluster/$(whoami)/miniconda/bin:$PATH"
```

### Pipenv

Tell Pipenv to install on the cluster FS, not in `/home`:

```bash
export WORKON_HOME=/cluster/$(whoami)/.python_cache
```

Common usage:

```bash
pipenv install tensorflow-gpu==1.12     # specific version
pipenv install --dev -e .               # install your project
pipenv install -r requirements.txt      # from requirements.txt
```

Match TensorFlow versions to the installed CUDA / cuDNN. Reproducibility across
Python runs is never guaranteed.

## Debugging

If your script fails and you want fast iteration, drop the GPU request (so you
don't queue) and add a small time limit:

```
#SBATCH --time=5
```

Or run interactively:

```bash
srun --pty --gres=gpu:1 bash -i
```

Don't sit in interactive mode for long. Other tips:

- Run with `python3 -i your_script.py` — it stops at the exception with a Python prompt.
- Use the `logging` module and tee to stdout — see <https://martinheinz.dev/blog/24>
  and <https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout>.

A reasonable Python skeleton (argparse + logging + a TOML config dump alongside
each output directory) helps reproducibility:

```python
import argparse, json, logging, os, sys, time
from os.path import join

def write_config_file(args, output_path, use_toml=False):
    if use_toml:
        import toml as json  # requires `pip3 install toml`
    filename = os.path.join(output_path, 'args.toml' if use_toml else 'args.json')
    with open(filename, 'w') as fp:
        try:
            import git  # requires `pip3 install pygit`
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            commit_message = repo.head.object.message
        except Exception:
            logging.warning('Not in a git repo — no commit info will be logged.')
            sha = '???'
            commit_message = '???'
        json.dump({
            'args': sys.argv,
            'git revision': {'message': commit_message, 'sha': sha},
            'arg.dict': args.__dict__,
        }, fp)
        logging.info('Wrote %s', filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', default='.')
    parser.add_argument('--output-folder', default='.')
    parser.add_argument('--output-tag', default='unnamed')
    args = parser.parse_args()

    args.output_tag += time.strftime('_%d.%m.%y_%H.%M.%S')
    args.output_folder = os.path.join(os.path.expanduser(args.output_folder), args.output_tag)
    os.makedirs(args.output_folder, exist_ok=True)

    logging.basicConfig(
        filename=join(args.output_folder, os.path.basename(__file__)[:-3] + '.log'),
        level=logging.DEBUG,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    write_config_file(args, args.output_folder, use_toml=True)

    my_awesome_script(args)
```

## Data layout

Two main areas for intermediate data:

- `/cluster` — distributed filesystem visible on every node (slower).
- `/scratch` — node-local SSD, faster but ephemeral.

Don't run heavy I/O against `/home` — it impairs everyone.

### `/scratch`

When a job starts, Slurm creates `/scratch/$SLURM_JOB_ID/` and removes it at the
end. Caveats:

1. World-readable/writable while it exists. Encrypt sensitive data.
2. No memory constraints — out-of-space failures are on you.
3. If multiple jobs need the same data on the same node, write your own
   coordination.

Per-node scratch sizes:

| Node    | Size  |
|---------|-------|
| lme49   | 3.5T  |
| lme50   | 867G  |
| lme51   | 3.6T  |
| lme52   | 1.6T  |
| lme53   | 410G  |
| lme170  | 371G  |
| lme171  | 371G  |
| lme221  | 867G  |
| lme222  | 867G  |
| lme223  | 867G  |

In a job script:

```bash
SCRATCH_CACHE=/scratch/$SLURM_JOB_ID
```

In Python:

```python
import os
scratch = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
```

### Important rules

- Put your data in a sub-directory named after your account
  (e.g. `/cluster/gropp` or `/scratch/gropp`).
- Delete it when you're done.
- There is **no backup** of cluster storage. Don't keep anything you can't
  afford to lose.
- Don't use `/home` for big data.

`/cluster` is also reachable from non-cluster Linux machines as `/net/cluster`,
and from Windows via WinSCP to `cluster.i5.informatik.uni-erlangen.de`.

## Output files

By default both stdout and stderr go to `slurm-%j.out` (where `%j` is the job
ID). Override with:

```
#SBATCH -o /home/%u/%x-%j-on-%N.out
#SBATCH -e /home/%u/%x-%j-on-%N.err
```

When something goes wrong, check these files first.

## Job control

```bash
squeue                            # running / queued jobs
scancel <jobid>                   # cancel one job
scancel -u <user>                 # cancel all your jobs
scancel -t PENDING -u <user>      # cancel only pending jobs
squeue -o "%8i %8u %15a %.10r %.10L %.5D %.10Q"   # show priority
```

## Highscore

Top GPU users since the start of the month:

```bash
sreport user top topcount=10 -t hourper --tres=gres/gpu \
    start=$(date --date="$(date +'%Y-%m-01')" +%D)
```

## Profiling

Slow job? Profile it.

### CPU: py-spy

Generate a flame graph (100s sample):

```bash
py-spy --flame profile.svg --duration 100 -- python3 matmul.py
```

### GPU: Nsight Systems

`nsys` is installed cluster-wide:

```bash
nsys profile -t nvtx,cuda,cudnn,cublas \
    --force-overwrite=true --stats=true --output=myapp \
    python3 ./train.py
```

Add NVTX ranges in your code so you can see which sections are slow. PyTorch:

```python
import torch.cuda.nvtx

with torch.cuda.nvtx.range("forward pass"):
    with torch.cuda.nvtx.range("encoder"):
        ...
    with torch.cuda.nvtx.range("decoder"):
        ...
```

TensorFlow: <https://github.com/NVIDIA/nvtx-plugins>.

Background:

- Talk: <https://www.youtube.com/watch?v=SpZ5MYRQc0U>
- Slides: <https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9339-profiling-deep-learning-networks.pdf>

## Advanced

### Email notifications

Slurm can email job-status changes — currently broken on this cluster.

### Time limit and dependencies

24-hour wall-time limit. Use `--dependency afterok:<jobid>` for follow-up jobs.
See <http://www.vrlab.umu.se/documentation/batchsystem/job-dependencies>.

### PyTorch

System PyTorch is installed for `python3`. For a different version, use a
virtualenv (see Pipenv above).

### Installing software

For Ubuntu packages, ask the admins.

For everything else, install under `/cluster/$(whoami)/opt` or
`/cluster/$(whoami)/local` and adjust your environment:

```bash
export PATH=/cluster/$(whoami)/local/bin:$PATH
export CPATH=/cluster/$(whoami)/local/include:$CPATH
export LD_LIBRARY_PATH=/cluster/$(whoami)/local/lib:$LD_LIBRARY_PATH
```

For CMake-based builds, set `CMAKE_INSTALL_PREFIX=/cluster/$(whoami)/local`.

### Analyse GPU usage on a specific node

```bash
srun --pty --nodelist=lme50 /opt/cluster/bin/nvtop
```
