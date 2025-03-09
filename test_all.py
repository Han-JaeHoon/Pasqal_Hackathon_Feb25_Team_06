from test_qaa import test_qaa
from test_qaoa import test_qaoa
import os

# 저장할 파일 경로 설정
save_dir = 'final_result'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

episode = 1
epoch = 1

for matrix_idx in range(500, 520):
    test_qaa(
      num_episode=episode,
      num_epoch=epoch,
      beta=25.0,
      lr=0.1,
      matrix_idx=matrix_idx,
      model_name="RL_QAA",
      save_dir=save_dir
  )

    test_qaa(
        num_episode=1,
        num_epoch=1,
        beta=100000000.0,
        matrix_idx=matrix_idx,
        lr=0.1,
        model_name="R_QAA",
        save_dir=save_dir,
    )

    test_qaoa(
        num_episode=episode,
        num_epoch=epoch,
        beta=25.0,
        lr=[0.1, 0.1],
        matrix_idx=matrix_idx,
        model_name="RL_QAOA",
        save_dir=save_dir,
    )
