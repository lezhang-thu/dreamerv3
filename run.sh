set -ex
GPU=0
env_name=montezuma_revenge
CUDA_VISIBLE_DEVICES=$GPU python dreamerv3/main.py \
	--logdir ~/logdir/dreamerv3/$env_name/v0 \
	--configs atari \
	--task atari_$env_name \
	--replay_context=0
