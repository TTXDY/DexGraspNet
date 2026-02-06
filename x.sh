python scripts/render_grasp_grid.py \
      --result_dir ../data/experiments/dexhand021_test/results \
      --objects "cylinder1,cylinder2,cylinder3,sphere2,cylinder5,cylinder6,cube1,cube2,cuboid1,cuboid2,cuboid3,sphere1" \
      --grasps 24 \
      --contact_links "01,02,03,04,05" \
      --output dexhand_grid2.html

 python main.py \
    --hand_model_type dexhand021 \
    --object_code_list "['cylinder1']" \
    --name dexhand021_test \
    --batch_size 24 \
    --n_iter 1000 \
    --gpu "0" \
    --contact_links "01" \
    --object_num_samples 2000 \
    --w_dis 100 \
    --w_pen 200

python main.py \
    --hand_model_type dexhand021 \
    --object_code_list "['cylinder1','cylinder2','cylinder3','sphere2','cylinder5','cylinder6','cube1','cube2','cuboid1','cuboid2','cuboid3','sphere1']" \
    --name dexhand021_grasping \
    --batch_size 24 \
    --n_iter 5000 \
    --gpu "0" \
    --contact_links "01,02,03,04" \
    --object_num_samples 2000 \
    --w_dis 100 \
    --w_pen 300 \
    --jitter_strength 0.05



python main.py \
    --hand_model_type dexhand021 \
    --object_code_list "['core-02773838-a1822be832091e036afa58a86636d6be','']" \
    --name dexhand021_grasping \
    --batch_size 24 \
    --n_iter 5000 \
    --gpu "0" \
    --n_contact 4 \
    --object_num_samples 2000 \
    --w_dis 100 \
    --w_pen 200 \
    --random_hand \
    --random_obj_scale

CUDA_VISIBLE_DEVICES=0 python scripts/generate_grasps.py \
    --all \
    --hand_model_type dexhand021 \
    --data_root_path ../data/meshdata \
    --result_path ../data/experiments/dexhand021_grasping \
    --batch_size_each 24 \
    --n_iter 5000 \
    --n_contact 4 \
    --object_num_samples 2000 \
    --w_dis 100 \
    --w_pen 300 \
    --random_hand \
    --random_obj_scale

 python visualize_result.py \
    --data_root_path ../data/meshdata_local \
    --hand_model_type dexhand021 \
    --object_code cube1 \
    --result_path ../data/experiments/dexhand021_rl_v3/results \
    --show_contact_points  \
    --num 0

 python tests/visualize_result.py \
    --data_root_path ../data/meshdata_local \
    --hand_model_type dexhand021 \
    --object_code cube2 \
    --result_path ../data/experiments/dexhand021_ground/results \
    --show_contact_points \
    --show_ground \
    --num 0

python main.py \
    --data_root_path ../data/meshdata_local \
    --hand_model_type dexhand021 \
    --object_code_list "['cylinder1']" \
    --name dexhand021_rl \
    --batch_size 24 \
    --n_iter 3000 \
    --gpu "0" \
    --contact_links "01,02,03,04" \
    --object_num_samples 2000 \
    --w_dis 100 \
    --w_pen 300



 CUDA_VISIBLE_DEVICES=0 python scripts/generate_grasps.py \
    --object_code_list cylinder1 cylinder2 \
    --hand_model_type dexhand021 \
    --data_root_path ../data/meshdata_local \
    --result_path ../data/experiments/dexhand021_grasping \
    --batch_size_each 32 \
    --n_iter 3000 \
    --n_contact 4 \
    --object_num_samples 1000 \
    --w_dis 100 \
    --w_pen 300 \
    --random_hand \
    --random_obj_scale \
    --overwrite


 CUDA_VISIBLE_DEVICES=0 python scripts/generate_grasps.py \
    --object_code_list core-02773838-4a1f62dbe8b091eabc49cae1a831a9e core-04074963-c99bbef1b2c514a81b22d29e47ec3f2 \
    --hand_model_type dexhand021 \
    --data_root_path ../data/meshdata \
    --result_path ../data/experiments/dexhand021_grasping \
    --batch_size_each 32 \
    --n_iter 3000 \
    --n_contact 4 \
    --object_num_samples 1000 \
    --w_dis 100 \
    --w_pen 300 \
    --random_hand \
    --random_obj_scale \
    --overwrite

 CUDA_VISIBLE_DEVICES=0 python scripts/generate_grasps.py \
    --todo \
    --hand_model_type dexhand021 \
    --data_root_path ../data/meshdata \
    --result_path ../data/experiments/dexhand021_randoms \
    --batch_size_each 32 \
    --n_iter 3000 \
    --n_contact 4 \
    --object_num_samples 2000 \
    --w_dis 100 \
    --w_pen 300 \
    --random_hand \
    --random_obj_scale


CUDA_VISIBLE_DEVICES=0 python scripts/generate_grasps.py \
    --data_root_path ../data/meshdata_local \
    --hand_model_type dexhand021 \
    --object_code_list  "['cylinder1','cylinder2','cylinder3','cylinder5','cylinder6','sphere1','cuboid2','cuboid3','cube1','cube2']" \
    --name dexhand021_rl_v2 \
    --batch_size_each 32 \
    --n_iter 5000 \
    --contact_links "01,02,03,04" \
    --object_num_samples 2000 \
    --w_dis 100 \
    --w_pen 500 \
    --max_e_pen 0.01

 python scripts/render_grasp_grid.py \
      --result_dir ../data/experiments/dexhand021_grasping \
      --objects "core-02773838-4a1f62dbe8b091eabc49cae1a831a9e, core-04074963-c99bbef1b2c514a81b22d29e47ec3f2" \
      --grasps 24 \
      --output dexhand_random1.html

 python scripts/render_grasp_grid.py \
    --result_dir ../data/experiments/dexhand021_rl_v2/results \
    --data_root_path ../data/meshdata_local \
    --contact_links "01,02,03,04" \
    --grasps 24 \
    --output dexhand_test_cylinder1.html

python main.py \
    --data_root_path ../data/meshdata_local \
    --hand_model_type dexhand021 \
    --object_code_list "['cylinder1','cylinder2','cylinder3','cylinder5','cylinder6','sphere1','cuboid2','cuboid3','cube1','cube2']" \
    --name dexhand021_rl_v3 \
    --batch_size 24 \
    --n_iter 3000 \
    --gpu "0" \
    --contact_links "01,02,03,04" \
    --object_num_samples 2000 \
    --w_dis 100 \
    --w_pen 500 \
    --max_e_pen 0.01
   
python main.py \
    --data_root_path ../data/meshdata_local \
    --hand_model_type dexhand021 \
    --object_code_list "['cube2']" \
    --name dexhand021_ground \
    --batch_size 24 \
    --n_iter 3000 \
    --gpu "0" \
    --contact_links "01" \
    --object_num_samples 2000 \
    --w_dis 100 \
    --w_pen 500 \
    --max_e_pen 0.01