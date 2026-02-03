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