# Evaluate CS-ARM-BN in a new source (S3 -> S8 and S8 -> S3)
CUBLAS_WORKSPACE_CONFIG=:16:8 \
python main.py \
  trainer=arm_bn \
  ++trainer.n_epochs=200 \
  ++trainer.optimizer.lr=0.001 \
  ++trainer.patience=20 \
  ++trainer.acc_steps=8 \
  ++trainer.scheduler=False \
  ++trainer.first_eval=True \
  ++trainer.model.auxiliar=False \
  ++trainer.distributed=False \
  ++trainer.full_domain_stats=False \
  \
  ++data.metadata_name=Metadata_Plate_ID \
  ++data.new_wells=False \
  ++data.preprocess=crop+flip \
  ++data.embeddings=False \
  ++data.channels=[0,1,2,3,4] \
  ++data.train_res=[256] \
  ++data.test_res=[256] \
  ++data.batch_size=64 \
  ++data.normalize=dataset \
  ++data.unlabeled=False \
  ++data.index_file=data/indices/all_poscons.pq \
  ++data.negative_index=data/indices/all_controls.pq \
  ++data.splitters.outer.n_splits=2 \
  ++data.splitters.outer.group=Metadata_Source \
  ++data.splitters.inner.group=Metadata_Batch \
  ++data.groups_per_batch=1 \
  ++data.groups_per_batch_eval=1 \
  ++data.distributed=False \
  ++data.source=False \
  ++data.neg_batch_size=128 \
  ++data.loader_with_controls=True \
  \
  ++wandb=True \
  ++trainer.wandb=True \
  ++models=False \
  ++distributed=False