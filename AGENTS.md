# Workspace Agent Instructions

## Cluster Execution Policy

- Always run Python workloads with `srun` (do not launch `python ...` directly for jobs).
- Use this exact order for Python workloads: (1) `srun` (2) `source ~/.bashrc` (3) `conda activate parr`.
- Execute steps (2) and (3) inside the `srun` job shell.
- Preferred Slurm partition: `alien`.
- Preferred Slurm QoS: `alien`.
- For long runs, avoid `node044`.

## Recommended command template

```bash
srun --partition=alien --qos=alien --exclude=node044 \
	bash -lc 'source ~/.bashrc && conda activate parr && python your_script.py [args...]'
```

## Notes

- Keep resource flags (`--gres`, `--mem`, `--time`, etc.) explicit per experiment.
- For short interactive checks, still keep `srun` as default unless explicitly overridden.
