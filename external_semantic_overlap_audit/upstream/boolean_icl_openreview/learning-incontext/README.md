<h2 align="center">
  Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions
</h2>


#### Dependencies

- Compatible with python 3
- Dependencies can be seen in `learning-incontext/requirements.txt`

#### Usage

Most of the relevant source code is in the `learning-incontext/src` directory. The set of command line arguments available can be seen in the respective `learning-incontext/src/args.py` file.

**Run Transformer on Conjunction**

At `learning-incontext/`:

```shell
$	python -m src.train -no-wandb -project in-context-learning -name conjunction_train -out_dir ./models/ -family san -model_name gpt2 -task conjunction -task_kwargs {} -data boolean -train_steps 25000 -no-analyze -curriculum_points_start 70 -curriculum_points_end 70 -num_tasks 0 -noise_rate 0.0 -n_positions 150 -n_dims 28 -n_embd 256 -n_layer 12 -n_head 8 -batch_size 64 -learning_rate 0.0001 -freeze 0 -gpu 1
```

**Run GPT-4 on Conjunction for direct evaluation**

At `learning-incontext/`:

```shell
$	python -m src.eval_openai -task conjunction -data boolean -n_dims 5 -n_points 50 -num_eval_examples 100 -prompt_type basic -model_type chat -model gpt-4
```
