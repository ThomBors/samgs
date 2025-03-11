mkdir -p ./trainlogs

#### MT10 ####

gamma=0.9
beta2=0.99
momentum=0.9
seed=40

export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt10 agent=samgs_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=$seed replay_buffer.batch_size=1280 agent.multitask.num_envs=10 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity > trainlogs/mt10_samgs_momentum$momentum\_gamma$gamma\_beta2$beta2\_sd$seed.log 2>&1 &

